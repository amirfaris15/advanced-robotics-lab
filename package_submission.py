"""Build a runnable coursework archive from the lab root."""
import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile


def build(output, video):
    root = Path(__file__).resolve().parent
    output, video = Path(output).resolve(), Path(video).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not video.is_file():
        raise FileNotFoundError('Supply the successful recorded video')
    with tempfile.TemporaryDirectory(prefix='cw2-submission-') as tmp:
        stage = Path(tmp)
        for source in root.glob('*.py'):
            shutil.copy2(source, stage/source.name)
        for name in ('models', 'utils'):
            shutil.copytree(root/name, stage/name,
                            ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
        for name in ('environment.yml', 'requirements.txt', 'cw2_solution.ipynb',
                     'SUBMISSION_README.md', 'LICENSE'):
            shutil.copy2(root/name, stage/name)
        evidence = stage/'artifacts'
        evidence.mkdir()
        for source in (root/'artifacts').glob('*.json'):
            shutil.copy2(source, evidence/source.name)
        # The course requires a <=300-word README, distinct from developer docs.
        shutil.copy2(root/'SUBMISSION_README.md', stage/'README.md')
        shutil.copy2(video, stage/'successful_run.mp4')
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(stage.rglob('*')):
                if source.is_file(): archive.write(source, source.relative_to(stage))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='STUDENT1NAME_STUDENT2NAME.zip')
    parser.add_argument('--video', required=True)
    args = parser.parse_args()
    print(build(args.output, args.video))
