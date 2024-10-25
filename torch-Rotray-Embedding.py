from __future__ import annotations
from math import pi, log

import torch
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from typing import Literal

# 헬퍼 함수들

def exists(val):
    # 주어진 값이 None이 아닌지 확인하는 함수
    # 반환: 값이 존재하면 True, 그렇지 않으면 False
    return val is not None

def default(val, d):
    # 주어진 값이 존재하면 그 값을 반환하고, 그렇지 않으면 기본값을 반환
    # 반환: val이 존재하면 val, 그렇지 않으면 d
    return val if exists(val) else d

# broadcat 사용되지 않음, tortoise-tts에서 사용 중인 함수
# def broadcat(tensors, dim = -1):
#     # 여러 텐서를 주어진 차원에서 브로드캐스트 후 연결
#     # 파라미터:
#     # - tensors: 연결할 텐서들의 리스트
#     # - dim: 연결할 차원 (기본값은 -1, 즉 마지막 차원)
#     # 반환: 브로드캐스트 후 연결된 텐서
#     broadcasted_tensors = broadcast_tensors(*tensors)
#     return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    '''
    파라미터
    t: 슬라이스할 텐서
    dim_slice: 슬라이스 객체
    dim: 슬라이스할 차원
    반환: 슬라이스된 텐서
    
    함수 동작 과정
    1. dim = -1 + 3 # t.ndim = 3, 따라서 dim = 2
    2. colons = [slice(None), slice(None), slice(None)]
    3. colons[2] = slice(None, 2) # 마지막 차원에 슬라이스 적용
    4. 최종 슬라이스 리스트: [slice(None), slice(None), slice(None, 2)]
    5. 슬라이싱 수행: t[:, :, :2]
    '''

    # dim이 음수 일 경우 t.ndim을 더해서 양수 인덱스로 변환
    # dim이 -2 이고 차원수가 5라면 dim은 3이 됨. 이는 음수 일 경우와 동일함
    dim += (t.ndim if dim < 0 else 0)
    #t.ndim = 3이면 colons = [slice(None), slice(None), slice(None)]가 됩니다.
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    # 리턴 값은 t[:, dim_slice, :] 와 동일
    return t[tuple(colons)]

# rotary embedding 헬퍼 함수들

def rotate_half(x):
    '''
    입력 텐서를 반으로 나누어 회전하는 함수로 동작 예시는 아래와 같음
    x = torch.tensor([[[1, 2, 3, 4, 5, 6]]])  # 크기: (1, 1, 6)
    
    rearrange(x, '... (d r) -> ... d r', r = 2) 적용 후:
    x = [[[[1, 2],
        [3, 4],
        [5, 6]]]]

    x1, x2 = x.unbind(dim = -1) 적용하여 마지막 차원을 기준으로 반으로 나눈 후:
    x1 = [[[1, 3, 5]]]
    x2 = [[[2, 4, 6]]]

    torch.stack((-x2, x1), dim = -1) 적용 후:
    [[[[-2, 1],
    [-4, 3],
    [-6, 5]]]]

    최종 리턴 값:
    x = [[[ -2,  1, -4,  3, -6,  5]]]
    '''

    # 만약 x의 크기가 (batch_size, seq_len, d * 2)라면 (batch_size, seq_len, d, 2)가 됩니다
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    # 텐서를 반으로 나누어 90도 회전. 이는 아래 t_transformed 에서 사인연산에 사용됨
    x = torch.stack((-x2, x1), dim = -1)
    # 텐서를 다시 원래의 크기로 변환 (batch_size, seq_len, d * 2)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    # 주어진 주파수와 텐서에 rotary embedding을 적용
    # 파라미터:
    # - freqs: 주파수 텐서
    # - t: 변환할 텐서
    # - start_index: 회전을 시작할 인덱스
    # - scale: 스케일링 팩터
    # - seq_dim: 시퀀스 차원
    # - freqs_seq_dim: 주파수 시퀀스 차원 (기본값은 None)
    # 반환: rotary embedding이 적용된 텐서
    dtype = t.dtype

    if t.ndim == 3 or exists(freqs_seq_dim):
        freqs_seq_dim = default(freqs_seq_dim, 0)
        # t.shape = (batch_size, seq_len, dim)이고 seq_dim이 -2이므로 t.shape[-2]은 seq_len을 추출함.
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # 텐서를 세 부분으로 나누기: 왼쪽, 중간(변환할 부분), 오른쪽
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # rotary embedding을 적용하되, 텐서를 제자리에서 수정하지 않음
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

# 학습된 회전 헬퍼이며 안쓰임
# def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
#     # 학습된 회전을 텐서에 적용
#     # 파라미터:
#     # - rotations: 회전 텐서
#     # - t: 변환할 텐서
#     # - start_index: 회전을 시작할 인덱스
#     # - freq_ranges: 주파수 범위 (기본값은 None)
#     # 반환: 회전이 적용된 텐서
#     if exists(freq_ranges):
#         rotations = einsum('..., f -> ... f', rotations, freq_ranges)
#         rotations = rearrange(rotations, '... r f -> ... (r f)')

#     rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
#     return apply_rotary_emb(rotations, t, start_index = start_index)

# 클래스들

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        # reddit 사용자 bloc97이 제안한 방법으로, fine-tuning 없이 더 긴 시퀀스 길이에 rotary embedding을 재조정
        # NTK 문헌과 연결이 있음
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        # theta 값을 차원에 따라 재조정
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        # 사용자 정의 주파수가 주어졌는지 확인
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # 언어 모델을 위한 주파수 계산
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            # 픽셀 데이터를 위한 주파수 계산
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            # 상수 주파수
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        # 주파수 캐시를 위한 버퍼 등록
        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.register_buffer('cached_freqs_seq_len', torch.tensor(0), persistent = False)

        # 주파수를 학습 가능한 파라미터로 설정
        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # 장치용 더미 텐서
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # 기본 시퀀스 차원 설정
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # 보간 계수 설정
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos 사용 여부
        self.use_xpos = use_xpos

        if not use_xpos:
            return

        # xpos 스케일 계산
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        # xpos 스케일 캐시를 위한 버퍼 등록
        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.register_buffer('cached_scales_seq_len', torch.tensor(0), persistent = False)

        # apply_rotary_emb를 정적 메서드로 추가
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        # 현재 장치를 반환
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        # 시퀀스 위치를 반환
        # 파라미터:
        # - seq_len: 시퀀스 길이
        # - device: 장치 정보
        # - dtype: 데이터 타입
        # - offset: 오프셋 값
        # 반환: 시퀀스 위치 텐서
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        # 쿼리 또는 키를 회전
        # 파라미터:
        # - t: 회전할 텐서
        # - seq_dim: 시퀀스 차원
        # - offset: 오프셋 값
        # - scale: 스케일링 팩터
        # 반환: 회전된 텐서
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), '길이 외삽 가능한 rotary embedding을 위해 `.rotate_queries_and_keys` 메서드를 사용하고 쿼리와 키를 모두 전달해야 합니다.'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

        freqs = self.forward(seq, seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        # 캐시된 키와 함께 쿼리를 회전
        # 파라미터:
        # - q: 쿼리 텐서
        # - k: 키 텐서
        # - seq_dim: 시퀀스 차원
        # - offset: 오프셋 값
        # 반환: 회전된 쿼리와 키 텐서
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype = dtype, device = device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        # 쿼리와 키를 함께 회전
        # 파라미터:
        # - q: 쿼리 텐서
        # - k: 키 텐서
        # - seq_dim: 시퀀스 차원
        # 반환: 회전된 쿼리와 키 텐서
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        # 스케일을 가져옴
        # 파라미터:
        # - t: 텐서
        # - seq_len: 시퀀스 길이
        # - offset: 오프셋 값
        # 반환: 스케일 텐서
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            exists(seq_len) and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales_seq_len.item()
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(seq_len)

        return scale

    def get_axial_freqs(self, *dims):
        # 축 주파수를 가져옴
        # 파라미터:
        # - dims: 각 차원의 크기
        # 반환: 축 주파수 텐서
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len = None,
        offset = 0
    ):
        # 주어진 텐서에 대해 주파수를 계산
        # 파라미터:
        # - t: 입력 텐서
        # - seq_len: 시퀀스 길이
        # - offset: 오프셋 값
        # 반환: 주파수 텐서
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(seq_len)

        return freqs
