"""Prepopulate the RemoteCLIP cache for offline inference."""

from rspd.remoteclip import resolve_checkpoint


if __name__ == "__main__":
    print(resolve_checkpoint("ViT-B-32", checkpoint_path=None))
