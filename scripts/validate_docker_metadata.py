"""Exercise Docker metadata-action v4 with the workflow's inputs and public repo data.

Clone docker/metadata-action v4 into results/validation/metadata-action first.
Requires Node, PyYAML, and existing gh authentication. Credentials are passed only
in the action subprocess environment; logs contain metadata, never token values.
No images are built or published. GitHub's event expressions are checked below.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import yaml

root = Path("results/validation").resolve()
action = root / "metadata-action/dist/index.js"
workflow = yaml.safe_load(Path(".github/workflows/docker.yml").read_text())
cases = [
    ("pull_request", "refs/pull/6/merge"),
    ("push", "refs/heads/main"),
    ("push", "refs/tags/v1.2.3"),
    ("workflow_dispatch", "refs/heads/main"),
]
results = []
for job_name, job in workflow["jobs"].items():
    if not job_name.startswith("build-"):
        continue
    meta = next(s for s in job["steps"] if s.get("id") == "meta")
    login = next(s for s in job["steps"] if s.get("uses", "").startswith("docker/login-action"))
    build = next(
        s for s in job["steps"] if s.get("uses", "").startswith("docker/build-push-action")
    )
    assert login["if"] == "github.event_name != 'pull_request'"
    assert build["with"]["push"] == "${{ github.event_name != 'pull_request' }}"
    for event, ref in cases:
        name = job_name + "-" + event + "-" + ref.replace("/", "_")
        case_dir = root / "metadata-cases" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "repository": {
                "default_branch": "main",
                "name": "H-JEPA",
                "owner": {"login": "jonwiggins"},
            },
            "number": 6,
            "pull_request": {
                "head": {
                    "ref": "optio/task-validation",
                    "sha": "b9da295f7fc8b268d44859957711d5228a6d4ca9",
                },
                "base": {"ref": "main"},
            },
        }
        (case_dir / "event.json").write_text(json.dumps(payload))
        for f in ["env", "output"]:
            (case_dir / f).write_text("")
        env = dict(
            PATH=os.environ["PATH"],
            HOME=os.environ["HOME"],
            GITHUB_REPOSITORY="jonwiggins/H-JEPA",
            GITHUB_EVENT_NAME=event,
            GITHUB_REF=ref,
            GITHUB_SHA="1d97f430000000000000000000000000000000000",
            GITHUB_EVENT_PATH=str(case_dir / "event.json"),
            GITHUB_ENV=str(case_dir / "env"),
            GITHUB_OUTPUT=str(case_dir / "output"),
            RUNNER_TEMP=str(case_dir),
            GITHUB_ACTOR="jonwiggins",
            GITHUB_WORKSPACE=str(Path.cwd()),
            GITHUB_RUN_ID="1",
            GITHUB_RUN_NUMBER="1",
            INPUT_IMAGES="ghcr.io/jonwiggins/h-jepa-" + job_name.removeprefix("build-"),
            INPUT_TAGS=meta["with"]["tags"],
        )
        env["INPUT_GITHUB-TOKEN"] = subprocess.check_output(
            ["gh", "auth", "token"], text=True
        ).strip()
        run = subprocess.run(
            ["node", str(action)], env=env, capture_output=True, text=True, timeout=60
        )
        (case_dir / "action.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError((name, run.stdout[-2000:], run.stderr[-2000:]))
        output = (case_dir / "output").read_text()
        match = re.search(r"^tags<<(.*)\n(.*?)\n\1$", output, re.M | re.S)
        assert match, output
        tags = match[2].splitlines()
        assert all(re.fullmatch(r"[a-z0-9./-]+:[\w][\w.-]{0,127}", t) for t in tags), tags
        assert any(t.endswith(":sha-1d97f43") for t in tags), tags
        if event == "pull_request":
            assert any(t.endswith(":pr-6") for t in tags)
            assert not any(t.endswith(":latest") or t.endswith(":main") for t in tags)
        elif ref == "refs/tags/v1.2.3":
            assert any(t.endswith(":1.2.3") for t in tags)
        else:
            assert any(t.endswith(":main") for t in tags)
        results.append(
            dict(
                job=job_name,
                event=event,
                ref=ref,
                tags=tags,
                login_and_push=event != "pull_request",
            )
        )
        print(name, tags, flush=True)
(root / "docker-metadata-results.json").write_text(json.dumps(results, indent=2) + "\n")
