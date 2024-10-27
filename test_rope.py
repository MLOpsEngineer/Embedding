import numpy as np
import torch
import pytest
from TorchRotrayEmbedding import RotaryEmbedding


def numpy_rotate_half(x):
    print("== NumPy rotate_half ==")
    print("입력 x:", x.shape)
    x = x.reshape(*x.shape[:-1], -1, 2)
    print("reshaped x:", x.shape)
    x1 = x[..., 0]
    x2 = x[..., 1]
    print("x1:", x1.shape)
    print("x2:", x2.shape)
    x = np.stack((-x2, x1), axis=-1)
    print("stacked x:", x.shape)
    x = x.reshape(*x.shape[:-2], -1)
    print("최종 x:", x.shape)
    return x


def numpy_apply_rotary_embedding(freqs, t, start_index=0, scale=1.0):
    print("== NumPy apply_rotary_embedding ==")
    print("입력 t:", t.shape)
    print("freqs:", freqs.shape)
    t_left = t[..., :start_index]
    end_index = start_index + freqs.shape[-1]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    print("t_left:", t_left.shape)
    print("t_middle:", t_middle.shape)
    print("t_right:", t_right.shape)
    t_transformed = (t_middle * np.cos(freqs) * scale) + (
        numpy_rotate_half(t_middle) * np.sin(freqs) * scale
    )
    print("t_transformed:", t_transformed.shape)
    result = np.concatenate((t_left, t_transformed, t_right), axis=-1)
    print("최종 결과:", result.shape)
    return result


class NumpyRotaryEmbedding:
    def __init__(self, dim, theta=10000):
        self.dim = dim
        self.theta = theta
        # PyTorch 구현과 동일하게 freqs 계산
        self.freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        print("NumPy freqs:", self.freqs.shape)

    def get_angles(self, seq_len, offset=0):
        print("== NumPy get_angles ==")
        positions = np.arange(seq_len, dtype=np.float32) + offset
        print("positions:", positions.shape)
        freqs = np.outer(positions, self.freqs)
        print("freqs before repeat:", freqs.shape)
        freqs = np.repeat(freqs, repeats=2, axis=-1)
        print("freqs after repeat:", freqs.shape)
        return freqs.astype(np.float32)

    def __call__(self, x, offset=0):
        print("== NumPy RotaryEmbedding 호출 ==")
        seq_len = x.shape[1]
        freqs = self.get_angles(seq_len, offset=offset)
        freqs = freqs[None, :, :]  # (1, seq_len, dim)
        print("freqs after adding batch dimension:", freqs.shape)
        return numpy_apply_rotary_embedding(freqs, x)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [8, 12, 16])
@pytest.mark.parametrize("dim", [32, 64, 128])
def test_rope(batch_size, seq_len, dim):
    # 입력 데이터 생성
    np.random.seed(0)
    torch.manual_seed(0)
    x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    x_torch = torch.from_numpy(x_np).float()

    print("입력 데이터 x_np:")
    print(x_np)

    # NumPy 기반 RoPE 적용
    numpy_rope = NumpyRotaryEmbedding(dim=dim)
    x_rotated_np = numpy_rope(x_np)
    print("NumPy 결과 x_rotated_np:", x_rotated_np.shape)
    print("NumPy 결과 x_rotated_np:")
    print(x_rotated_np)

    # PyTorch 기반 RoPE 적용
    torch_rope = RotaryEmbedding(dim=dim)
    x_rotated_torch = torch_rope.rotate_queries_or_keys(x_torch)
    x_rotated_torch = x_rotated_torch.detach().numpy()
    print("PyTorch 결과 x_rotated_torch:", x_rotated_torch.shape)
    print("PyTorch 결과 x_rotated_torch:")
    print(x_rotated_torch)

    # 결과 비교
    difference = np.max(np.abs(x_rotated_np - x_rotated_torch))
    print(f"두 구현 간의 최대 차이: {difference}")
    print("NumPy와 PyTorch 구현 결과 차이 벡터:")
    print(x_rotated_np - x_rotated_torch)
    assert np.allclose(
        x_rotated_np, x_rotated_torch, atol=1e-6
    ), f"NumPy와 PyTorch 구현 결과가 일치하지 않습니다. 최대 차이: {difference}"

    print("NumPy와 PyTorch 구현 결과가 일치합니다.")


if __name__ == "__main__":
    test_rope(1, 128, 32)
