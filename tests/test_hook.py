"""False-positive suite for the PreToolUse hook. False positives (blocking a
harmless command) are worse than misses, so ALLOWED is the longer list."""

import io
import json

import pytest

from gpushift import hook

BLOCKED = [
    # (command, expected preset suggestion)
    ("python train.py", "one"),
    ("python3 train_resnet.py --epochs 10", "one"),
    ("python -u scripts/train.py", "one"),
    ("python src/finetune_llama.py --lr 1e-4", "one"),
    ("python resnet_train.py", "one"),
    ("python pretrain.py", "one"),
    ("python -m src.train", "one"),
    ("python -m my_project.training.sft", "one"),
    ("torchrun --nproc_per_node=2 train.py", "two"),
    ("torchrun --nproc-per-node 1 main.py", "one"),
    ("deepspeed main.py --deepspeed ds.json", "one"),
    ("accelerate launch --multi_gpu main.py", "two"),
    ("accelerate launch --num_processes 2 main.py", "two"),
    ("accelerate launch main.py", "one"),
    ("python -m torch.distributed.run --nproc_per_node 2 main.py", "two"),
    ("python -m torch.distributed.launch main.py", "one"),
    ("CUDA_VISIBLE_DEVICES=0 python main.py", "one"),
    ("CUDA_VISIBLE_DEVICES=1 python -m mypkg.eval", "one"),
    ("export CUDA_VISIBLE_DEVICES=0; python main.py", None),  # export form is not detected (documented miss)
    ("cd exp && python train.py --cfg a.yaml", "one"),
    ("source .venv/bin/activate && python train.py", "one"),
    ("nohup python train.py > log.txt 2>&1 &", "one"),
    ("uv run python train.py", "one"),
    ("uv run --with torch train.py", None),  # uv runs the script directly; not python-prefixed (miss is OK)
    ("conda run -n ml python train.py", "one"),
    ("poetry run torchrun --nproc_per_node 2 main.py", "two"),
    (r"C:\envs\ml\python.exe train.py", "one"),
    (".venv/Scripts/python.exe train.py", "one"),
    ('$env:CUDA_VISIBLE_DEVICES = "0"; python main.py', "one"),
    ("& python train.py", "one"),
    ("time python train.py", "one"),
]

ALLOWED = [
    'python -c "import torch; print(torch.cuda.is_available())"',
    "python train.py --help",
    "python train.py -h",
    "python -m pytest tests/test_train.py",
    "pytest -k train",
    "python -m pip install torch",
    "pip install transformers",
    "python setup.py develop",
    "python main.py",
    "python eval.py --ckpt last.pt",
    "python constrain.py",
    "python strain_gauge.py",
    "python train.py --dry-run",
    "python train.py # no-queue",
    "GPUSHIFT_NO_QUEUE=1 python train.py",
    "gpushift submit --one -- python train.py",
    "gpushift submit --two -- torchrun --nproc_per_node 2 train.py",
    'gpushift submit --one -- "python train.py && python eval.py"',
    "gpushift status --json",
    "gpushift logs 12 -f",
    "ts -G 1 python train.py",
    "cat train.py",
    "grep -r train src/",
    "ls train/",
    "git commit -m 'train.py: fix lr'",
    "echo python train.py",
    "nvidia-smi",
    "python -m venv .venv",
    "python",
    "vim train.py",
    "black train.py",
    "ruff check src/train.py",
    "python -m mypy train.py",
    "sed -n 1,20p train.py",
    "head -50 logs/train.log",
    "tail -f runs/train/log.txt",
    "python tools/plot_training_curves.py",
    "python scripts/prepare_data.py",
    "python -m json.tool config.json",
    "torchrun --help",
    "accelerate config",
    "accelerate env",
    "deepspeed --version",
    "",
    "   ",
    "python 'unterminated",
]


@pytest.mark.parametrize("cmd,preset", BLOCKED)
def test_blocked(cmd, preset):
    v = hook.check(cmd)
    if preset is None:
        return  # documented misses: only make sure nothing crashes
    assert v.block, cmd
    assert v.preset == preset, (cmd, v)


@pytest.mark.parametrize("cmd", ALLOWED)
def test_allowed(cmd):
    assert not hook.check(cmd).block, cmd


def test_free_is_user_only():
    assert hook.check("gpushift free").block
    assert hook.check("gpushift free --soft").block
    assert hook.check("cd x && gpushift resume").block
    assert not hook.check("GPUSHIFT_BY_USER=1 gpushift free").block
    # reading the help is fine, and doesn't take the rest of the line down with it
    assert not hook.check("gpushift free --help").block
    assert not hook.check("gpushift status 2>&1; echo ---; gpushift free --help 2>&1").block
    assert not hook.check("gpushift resume -h").block
    assert hook.check("gpushift free --help; gpushift free").block
    assert hook.check("gpushift free --soft && echo --help").block
    assert not hook.check("gpushift status").block
    assert not hook.check("gpushift submit --one -- python free.py").block
    # Only mentioned, not run: commit messages, echo, grep.
    assert not hook.check('git commit -m "Explain gpushift free and gpushift resume"').block
    assert not hook.check('git commit -m "Add docs\n\ngpushift free checkpoints jobs."').block
    assert not hook.check('echo "run gpushift resume when you are back"').block
    assert not hook.check('grep -rn "gpushift free" README.md').block
    assert hook.check("C:/tools/gpushift.exe free").block
    assert hook.check("$env:GPUSHIFT_TICK = 1; gpushift resume").block
    assert not hook.check("$env:GPUSHIFT_BY_USER = 1; gpushift resume").block


def test_config_allow_deny():
    assert not hook.check("python train.py", allow=[r"^python train\.py$"]).block
    assert hook.check("python my_thing.py", deny=["my_thing"]).block


def run_main(monkeypatch, capsys, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload) if not isinstance(payload, str) else payload))
    rc = hook.main()
    return rc, capsys.readouterr().err


def test_main_protocol(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("GPUSHIFT_HOME", str(tmp_path))
    rc, err = run_main(monkeypatch, capsys, {"tool_name": "Bash", "tool_input": {"command": "python train.py"}})
    assert rc == 2 and "gpushift submit --one" in err and "python train.py" in err
    rc, _ = run_main(monkeypatch, capsys, {"tool_name": "PowerShell", "tool_input": {"command": "torchrun x.py"}})
    assert rc == 2
    rc, _ = run_main(monkeypatch, capsys, {"tool_name": "Read", "tool_input": {"file_path": "train.py"}})
    assert rc == 0
    rc, _ = run_main(monkeypatch, capsys, "not json")
    assert rc == 0
    (tmp_path / "config.toml").write_text("[hook]\nenabled = false\n")
    rc, _ = run_main(monkeypatch, capsys, {"tool_name": "Bash", "tool_input": {"command": "python train.py"}})
    assert rc == 0
    (tmp_path / "config.toml").write_text("this is [ not toml")
    rc, _ = run_main(monkeypatch, capsys, {"tool_name": "Bash", "tool_input": {"command": "python train.py"}})
    assert rc == 2  # broken config: hook still works with defaults
