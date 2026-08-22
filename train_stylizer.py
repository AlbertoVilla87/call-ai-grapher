import argparse
import logging

from call_ai_grapher.models import train_stylizer


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the pix2pix character stylizer (ugly -> pretty)")
    parser.add_argument("--data", default="dataset/alphabet", help="alphabet dataset directory (one dir per class)")
    parser.add_argument("--out", default="models/char_stylizer.pt", help="where to write the checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="pairs per optimization step")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate for generator and discriminator")
    parser.add_argument("--variants", type=int, default=20, help="degraded variants generated per glyph")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args()
        metrics = train_stylizer(
            args.data,
            args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            variants_per_glyph=args.variants,
            seed=args.seed,
        )
        logging.info(
            "Done: final L1 %.4f, gan %.4f, discriminator %.4f",
            metrics["gen_l1"],
            metrics["gen_gan"],
            metrics["disc"],
        )
    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
