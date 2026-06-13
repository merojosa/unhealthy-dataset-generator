from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Esto se mete recursivamente en el folder, y retorna
# una lista ordenada de rutas de imágenes con alguna extensión válida
# ordenadas lexicográficamente
def load_image_paths(folder: str) -> list[Path]:
    folder_path = Path(folder)
    # Por si el path está mal puesto
    if not folder_path.exists():
        raise FileNotFoundError(f"La carpeta no existe: {folder}")
    # Retorne todas las rutas de imágenes en ese folder lexicográficamente ordenadas
    return sorted(
        path for path in folder_path.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

# El objetivo de esto es manejar todas las imágenes con formato RGB
def load_image(path: Path) -> Image.Image | None:
    try:
        # Lo que va a hacer img.convert("RGB") es,
        # a cada pixel de la imagen, asignarle un valor RGB (tres sequential bytes)
        # y retornar el "vector" de RGBs (el equivalente a un vector en img)
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as error:
        print(f"Opa, no pude leer {path}: {error}")
        return None

# La idea de los perceptual hashes es que dos imágenes "parecidas", 
# le tiren hashes similares. 
# Ahora, cuando uno hace difference hash, usted lo que hace es comparar
# los cambios de brightness entre pixeles adyacentes 
# (dos pixeles son adyacentes sii uno está a la izquierda del otro)
# Se usa perceptual_dHash porque es como el punto medio entre effectiveness y speed.
def perceptual_dhash(image: Image.Image, hash_size: int = 8) -> int:

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    # OJO: el perceptual_dhash siempre trabaja con imágenes en escala de grises
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        resample_filter,
    )

    pixels = np.asarray(gray, dtype=np.int16)

    # Compara cada pixel con el de la derecha.
    differences = pixels[:, 1:] > pixels[:, :-1]

    hash_value = 0

    for bit in differences.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value

# Esto solo me dice cuántos bits son diferentes entre dos hashes
# Ojo que eso (claramente) lo puedo sacar con un XOR
def hamming_distance(hash_1: int, hash_2: int) -> int:
    return (hash_1 ^ hash_2).bit_count()

# Esto busca, para cada imagen: 
# 1. Cárguela
# 2. Calcule el hash perceptual 
# 3. Calcule embedding CLIP
# y de paso descarta imágenes si por A o por B no se pudieron abrir
# Va a retornar una tupla de (valid_rutas, embeddings, hashes)
def compute_embeddings_and_hashes_from_paths(
    model,
    image_paths: list[Path],
    batch_size: int = 32,
    hash_size: int = 8,
) -> tuple[list[Path], np.ndarray, list[int]]:

    valid_paths = []
    all_embeddings = []
    all_hashes = []
    # las imágenes se van a ir procesando en batches de tamagno 32
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Calculando embeddings y hashes"):
        # saque las rutas del batch actual
        batch_paths = image_paths[start:start + batch_size]

        images = []
        paths_that_loaded = []
        hashes_that_loaded = []

        for path in batch_paths:
            image = load_image(path)

            if image is not None:
                image_hash = perceptual_dhash(image, hash_size=hash_size)

                images.append(image)
                paths_that_loaded.append(path)
                hashes_that_loaded.append(image_hash)

        if len(images) == 0:
            continue
        # yo voy a normalizar los vectores para que el producto punto de cualesquiera dos sea 
        # la similitud coseno
        embeddings = model.encode(
            images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_embeddings.append(embeddings.astype(np.float32))
        valid_paths.extend(paths_that_loaded)
        all_hashes.extend(hashes_that_loaded)

        for image in images:
            image.close()

    if len(all_embeddings) == 0:
        return [], np.empty((0, 0), dtype=np.float32), []

    return valid_paths, np.vstack(all_embeddings), all_hashes

# Esto es solo para ir escribiendo resultados en un csv
# Si es la primera vez que se escribe en el csv, se mete un header, sino no
def append_rows_to_csv(
    rows: list[dict],
    columns: list[str],
    output: str,
    first_write: bool,
) -> bool:

    if not rows:
        return first_write

    output_path = Path(output)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(
        output,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )

    return False

# Esto es solo para crear un csv vacío por si no se encuentra nada con 
# la similitud esperada
def create_empty_csv(output: str, columns: list[str]) -> None:
    output_path = Path(output)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=columns).to_csv(output, index=False)

# Ok, esto es para comparar una imagen con todas las posteriores
def find_similar_pairs_three_methods_chunked(
    image_paths: list[Path],
    embeddings: np.ndarray,
    hashes: list[int],
    clip_threshold: float,
    hash_max_distance: int,
    combined_threshold: float,
    clip_weight: float,
    clip_output: str,
    hash_output: str,
    combined_output: str,
    chunk_size: int = 512,
    hash_size: int = 8,
) -> None:

    n = len(image_paths)
    hash_bits = hash_size * hash_size
    # van a haber tres comparaciones: CLIP, HASH, y el combinado
    
    clip_columns = [
        "image_1",
        "image_2",
        "similarity",
        "equal",
    ]

    hash_columns = [
        "image_1",
        "image_2",
        "hash_distance",
        "hash_similarity",
        "equal",
    ]

    combined_columns = [
        "image_1",
        "image_2",
        "clip_similarity",
        "hash_distance",
        "hash_similarity",
        "combined_score",
        "equal",
    ]

    first_clip_write = True
    first_hash_write = True
    first_combined_write = True

    for start in tqdm(range(0, n, chunk_size), desc="Comparando imágenes"):
        end = min(start + chunk_size, n)

        # Como los embeddings están normalizados, el producto punto es cosine similarity.
        sim_block = embeddings[start:end] @ embeddings.T

        clip_rows = []
        hash_rows = []
        combined_rows = []

        for local_i in range(end - start):
            i = start + local_i

            # Solo revisamos j > i para no repetir pares ni comparar una imagen consigo misma.
            for j in range(i + 1, n):
                clip_similarity = float(sim_block[local_i, j])

                distance = hamming_distance(hashes[i], hashes[j])
                hash_similarity = 1.0 - (distance / hash_bits)
                # el combined score como corresponde
                combined_score = (
                    clip_weight * clip_similarity
                    + (1.0 - clip_weight) * hash_similarity
                )
                
                if clip_similarity >= clip_threshold:
                    clip_rows.append({
                        "image_1": str(image_paths[i]),
                        "image_2": str(image_paths[j]),
                        "similarity": clip_similarity,
                        "equal": True,
                    })

                if distance <= hash_max_distance:
                    hash_rows.append({
                        "image_1": str(image_paths[i]),
                        "image_2": str(image_paths[j]),
                        "hash_distance": distance,
                        "hash_similarity": hash_similarity,
                        "equal": True,
                    })

                if combined_score >= combined_threshold:
                    combined_rows.append({
                        "image_1": str(image_paths[i]),
                        "image_2": str(image_paths[j]),
                        "clip_similarity": clip_similarity,
                        "hash_distance": distance,
                        "hash_similarity": hash_similarity,
                        "combined_score": combined_score,
                        "equal": True,
                    })

        first_clip_write = append_rows_to_csv(
            clip_rows,
            clip_columns,
            clip_output,
            first_clip_write,
        )

        first_hash_write = append_rows_to_csv(
            hash_rows,
            hash_columns,
            hash_output,
            first_hash_write,
        )

        first_combined_write = append_rows_to_csv(
            combined_rows,
            combined_columns,
            combined_output,
            first_combined_write,
        )
    # first_clip_write va a ser true sii no pudo append nada
    # lo mismo con los otros
    if first_clip_write:
        create_empty_csv(clip_output, clip_columns)

    if first_hash_write:
        create_empty_csv(hash_output, hash_columns)

    if first_combined_write:
        create_empty_csv(combined_output, combined_columns)

# esto es solo para definir el formato correcto del file del output
def output_name_from_prefix(prefix: str, suffix: str) -> str:
    path = Path(prefix)

    if path.suffix == ".csv":
        path = path.with_suffix("")

    return str(path.with_name(path.name + suffix))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("folder", help="Carpeta con imágenes")
    # CLIP
    # OJO: el default es el min umbral de similitud
    parser.add_argument("--clip-threshold", type=float, default=0.95)

    # Hash perceptual
    # en este caso la max hamming distance permitida va a ser 5
    parser.add_argument("--hash-max-distance", type=int, default=5)
    parser.add_argument("--hash-size", type=int, default=8)

    # Combinación
    # por ahora le da 70% peso a CLIP, 30% a hash
    parser.add_argument("--combined-threshold", type=float, default=0.94)
    parser.add_argument("--clip-weight", type=float, default=0.70)

    # Outputs
    parser.add_argument("--output-prefix", default="similarity_report")
    parser.add_argument("--clip-output", default=None)
    parser.add_argument("--hash-output", default=None)
    parser.add_argument("--combined-output", default=None)

    # Performance
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)

    args = parser.parse_args()
    # vea que el peso sea una proporción válida
    if not (0.0 <= args.clip_weight <= 1.0):
        raise ValueError("--clip-weight debe estar entre 0 y 1.")

    image_paths = load_image_paths(args.folder)
    print(f"Encontré {len(image_paths)} imágenes.")
    # vamos a usar la versión más reciente de CLIP, que puntualmente es bueno con
    # similitudes de imágenes
    model = SentenceTransformer("clip-ViT-B-32")

    image_paths, embeddings, hashes = compute_embeddings_and_hashes_from_paths(
        model,
        image_paths,
        batch_size=args.batch_size,
        hash_size=args.hash_size,
    )

    print(f"Se pudieron procesar {len(image_paths)} imágenes.")
    print(f"Shape de embeddings: {embeddings.shape}")
    print(f"Se calcularon {len(hashes)} hashes perceptuales.")

    if len(image_paths) == 0:
        print("No se pudo cargar ninguna imagen.")
        return

    clip_output = args.clip_output or output_name_from_prefix(
        args.output_prefix,
        "_clip.csv",
    )

    hash_output = args.hash_output or output_name_from_prefix(
        args.output_prefix,
        "_hash.csv",
    )

    combined_output = args.combined_output or output_name_from_prefix(
        args.output_prefix,
        "_combined.csv",
    )

    find_similar_pairs_three_methods_chunked(
        image_paths=image_paths,
        embeddings=embeddings,
        hashes=hashes,
        clip_threshold=args.clip_threshold,
        hash_max_distance=args.hash_max_distance,
        combined_threshold=args.combined_threshold,
        clip_weight=args.clip_weight,
        clip_output=clip_output,
        hash_output=hash_output,
        combined_output=combined_output,
        chunk_size=args.chunk_size,
        hash_size=args.hash_size,
    )

    print(f"Reporte CLIP guardado en: {clip_output}")
    print(f"Reporte hash guardado en: {hash_output}")
    print(f"Reporte combinado guardado en: {combined_output}")


if __name__ == "__main__":
    main()