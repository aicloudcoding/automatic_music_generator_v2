"""Report whether TensorFlow can see a GPU.  Run: python -m amg.check_gpu"""

from __future__ import annotations


def gpu_summary() -> list[str]:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    names = []
    for g in gpus:
        details = tf.config.experimental.get_device_details(g)
        names.append(details.get("device_name", g.name))
    return names


def main() -> None:
    import tensorflow as tf

    print("TensorFlow", tf.__version__, "| built with CUDA:", tf.test.is_built_with_cuda())
    gpus = gpu_summary()
    if gpus:
        print("GPU(s) found:", ", ".join(gpus))
    else:
        print(
            "No GPU found - training will run on the CPU.\n"
            "On Windows, TensorFlow only uses the GPU inside WSL2: "
            'pip install "tensorflow[and-cuda]" there, and check `nvidia-smi` works.'
        )


if __name__ == "__main__":
    main()
