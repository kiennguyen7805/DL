"""Kaggle CPU cell: test a tiny HF upload without touching model artifacts."""
import hashlib
import json
import re
import uuid
from pathlib import Path

REPO_ID = 'iraprp/deeplearning'
SECRET_NAME = 'HF_TOKEN2'


def error_detail(error, token):
    response = getattr(error, 'response', None)
    message = getattr(error, 'server_message', None)
    if not message and response is not None:
        try:
            body = response.json()
            if isinstance(body, dict):
                message = body.get('error') or body.get('message')
        except (ValueError, AttributeError):
            pass
    message = str(message or error)
    if token:
        message = message.replace(token, '[REDACTED]')
    message = re.sub(r'hf_[A-Za-z0-9_]+', '[REDACTED]', message)
    message = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', message, flags=re.I)
    message = re.sub(r'https?://[^\s]+', '[URL REDACTED]', message)
    return {'error_type': type(error).__name__,
            'http_status': getattr(response, 'status_code', None),
            'server_message': message[:2000]}


def diagnose(api, token, download, output_dir):
    """Dependency injection allows local tests with an offline fake HF client."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'repository': REPO_ID, 'status': 'FAIL', 'stage': 'authentication',
              'training': 'NOT RUN', 'checkpoint_files': 'NOT ACCESSED'}
    try:
        api.whoami()  # Do not print token, user profile, or the response.
        report['authentication'] = 'PASS'
        report['stage'] = 'repository_detection'
        found = []
        errors = {}
        for kind in ('dataset', 'model'):
            try:
                api.repo_info(repo_id=REPO_ID, repo_type=kind)
                found.append(kind)
            except Exception as error:
                errors[kind] = error_detail(error, token)
        if len(found) != 1:
            report['repository_errors'] = errors
            raise RuntimeError('Expected exactly one accessible dataset/model repository; found ' + str(found))
        kind = found[0]
        report['repo_type'] = kind
        name = 'upload_diagnostics/probe_' + uuid.uuid4().hex + '.txt'
        payload = b'ROD HF diagnostic: small upload only; no checkpoint or training.\n'
        report.update(stage='upload_small_file', remote_probe_path=name, bytes=len(payload))
        print('Testing tiny upload:', REPO_ID, kind, name, flush=True)
        api.upload_file(path_or_fileobj=payload, path_in_repo=name, repo_id=REPO_ID,
                        repo_type=kind, token=token, commit_message='ROD: diagnostic tiny-file upload')
        report['upload_small_file'] = 'PASS'
        report['stage'] = 'download_checksum'
        path = download(repo_id=REPO_ID, repo_type=kind, filename=name,
                        token=token, local_dir=str(output/'download'))
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != hashlib.sha256(payload).hexdigest():
            raise RuntimeError('Downloaded probe SHA256 mismatch')
        report.update(status='PASS', stage='complete', download_checksum='PASS', sha256=actual)
        report['scope'] = 'Small-file upload/download only; checkpoint-size and upload_folder still NOT VERIFIED'
    except Exception as error:
        report.update(error_detail(error, token))
    (output/'hf_upload_diagnostic.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('Share this JSON output. Do not share your token.', flush=True)
    return report


def main():
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import HfApi, hf_hub_download
    token = UserSecretsClient().get_secret(SECRET_NAME)
    if not token or not token.strip():
        raise RuntimeError('Missing Kaggle Secret HF_TOKEN2')
    return diagnose(HfApi(token=token), token, hf_hub_download,
                    '/kaggle/working/hf_upload_diagnostic')


if __name__ == '__main__':
    main()
