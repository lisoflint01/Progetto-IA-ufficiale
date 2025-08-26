# main_augment.py
import argparse
from pathlib import Path

# importa le funzioni dal file a fianco
from augment_train import (
    count_real_images,
    compute_s,
    synthesize_images_inplace,
)

def find_class_dirs(train_dir: Path):
    """Ritorna solo le sottocartelle (classi) dentro train_dir."""
    return [p for p in train_dir.iterdir() if p.is_dir()]

def main():
    parser = argparse.ArgumentParser(
        description="Augment in-place delle immagini per ogni classe (sottocartella) dentro la cartella train."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path(r"C:\Users\liso4\miniconda3\IA\Progetto2\dataset\train"),
        help="Percorso alla cartella 'train' che contiene le classi come sottocartelle."
    )
    parser.add_argument("--ratio", type=float, default=0.25, help="Rapporto S/R da generare.")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la riproducibilità.")
    parser.add_argument("--classes", nargs="*", default=None, help="Opzionale: classi da processare (es. cat dog).")

    args = parser.parse_args()

    train_dir: Path = args.train_dir
    if not train_dir.exists() or not train_dir.is_dir():
        raise SystemExit(f"Train dir non valida: {train_dir}")

    class_dirs = find_class_dirs(train_dir)
    if args.classes:
        names = set(args.classes)
        class_dirs = [d for d in class_dirs if d.name in names]
        if not class_dirs:
            raise SystemExit(f"Nessuna classe trovata tra: {', '.join(args.classes)}")

    print(f"[INFO] Train dir: {train_dir}")
    print(f"[INFO] Classi da processare: {', '.join(d.name for d in class_dirs)}")
    print(f"[INFO] ratio={args.ratio}  seed={args.seed}")

    total_made = 0
    for class_dir in class_dirs:
        R = count_real_images(class_dir)
        if R == 0:
            print(f"[WARN] Nessuna immagine reale trovata in '{class_dir.name}', salto.")
            continue

        S = compute_s(R, args.ratio)
        print(f"[CLASS] {class_dir.name}: R={R} -> genero S={S} immagini sintetiche...")
        made = synthesize_images_inplace(class_dir, S, seed=args.seed)
        total_made += made
        print(f"[DONE]  {class_dir.name}: create {made} immagini.")

    print(f"[SUMMARY] Totale immagini sintetiche create: {total_made}")

if __name__ == "__main__":
    main()
