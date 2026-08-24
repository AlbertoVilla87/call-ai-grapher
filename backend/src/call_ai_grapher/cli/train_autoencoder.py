import argparse
import logging

from call_ai_grapher.models import train_autoencoder


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the character autoencoder for latent-space blending")
    parser.add_argument("--data", default="dataset/alphabet", help="alphabet dataset directory (one dir per class)")
    parser.add_argument("--out", default="models/char_autoencoder.pt", help="where to write the checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="samples per optimization step")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--variants", type=int, default=10, help="degraded variants generated per glyph")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args(argv)
        metrics = train_autoencoder(
            args.data,
            args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            variants_per_glyph=args.variants,
            seed=args.seed,
        )
        logging.info("Done: final reconstruction L1 %.4f", metrics["recon_l1"])
    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    main()
