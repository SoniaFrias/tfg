import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

N_CELLS_PER_SAMPLE = 5000
MARKERS_BLAST = ['SSC-A','CD45','CD34','CD117']
MARKERS_TUBE1 = ['FSC-A','SSC-A','CD7','CD56','CD34','CD117','CD33','HLA-DR','CD13','CD45']
MARKERS_TUBE2 = ['FSC-A','SSC-A','CD15','CD22','CD34','CD117','CD19','HLA-DR','CD13','CD45']
MARKERS_TUBE3 = ['FSC-A','SSC-A','CD36','CD14','CD34','CD117','CD11b','HLA-DR','CD13','CD45']
MARKERS_TUBE4 = ['FSC-A','SSC-A','CD2','CD133','CD34','CD117','CD33','HLA-DR','CD13','CD45']
LABEL_BLAST = ['Blast']
LABEL_LAIP = ['Blast', 'LAIP']

def split_from(directory, test_ratio, validation_ratio, tube=None): 
  """
  Divide los ficheros de sample_info.csv en tres dataframes para entrenamiento, 
  validación y test, sin separar las muestras de un mismo paciente entre los conjuntos 

  Args:
    directory (Path): Ruta del directorio de la base de datos correspondiente
    test_ratio (float): Proporción de datos de test sobre el conjunto de muestras únicas 
    validation_ratio (float): Propoción de datos de validación sobre el conjunto restante anterior 
    tube (str): Utilizado si solo se tiene en cuenta un Tubo

  Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (df_train, df_val, df_test)
  """
  data = pd.read_csv(directory / 'sample_info.csv')
  file_ids = data[f'{directory.name}_ID'].str.split('_')
  data['patient_id'] = file_ids.str[1]
  if tube is not None:
    data['tube'] = file_ids.str[3]
    data = data[data['tube'] == tube]
  df_unique = data.drop_duplicates(subset=['patient_id'], keep='first')

  df_train_val, df_test = train_test_split(
    df_unique,
    test_size=test_ratio,
    random_state=42,
    shuffle=True,
    stratify=df_unique['sample_type']
  )

  df_train, df_validation = train_test_split(
    df_train_val,
    test_size=validation_ratio,
    random_state=42,
    shuffle=True,
    stratify=df_train_val['sample_type']
  )

  # generalizar a todos los ficheros de un mismo paciente
  df_train = data[data['patient_id'].isin(df_train['patient_id'])]
  df_validation = data[data['patient_id'].isin(df_validation['patient_id'])]
  df_test =  data[data['patient_id'].isin(df_test['patient_id'])]
  
  print(len(df_train), len(df_validation), len(df_test))

  return df_train, df_validation, df_test


def read_cells(db_path, sample_id, markers, label):
  """
  Carga los datos de citometría para unos marcadores concretos, estandarizando 
  los nombres de los canales para eliminar sufijos de fluoróforos garantizando el orden
  y, además, su etiqueta correspondiente

  Args:

  Returns:
    
  """ 
  data = pd.read_csv(db_path / 'CSV' / f'{sample_id}.csv', usecols=lambda name: any(m in name for m in markers))
  data.columns = [next(name for name in markers if name in col) for col in data.columns]
  data = data[markers]
  labels = pd.read_csv(db_path / 'labels' / f'{sample_id}.csv', usecols=label)
  return data, labels

def load_blast_sample(db_path, sample_id, markers):
  data, labels = read_cells(db_path, sample_id, markers, LABEL_BLAST)
  return data, labels
 
def load_laip_sample(db_path, sample_id, markers):
  data, labels = read_cells(db_path, sample_id, markers, LABEL_LAIP)
  mask = labels[LABEL_BLAST[0]] == 1
  labels = labels.drop(columns=LABEL_BLAST[0])
  idx = labels[mask].index
  return data.loc[idx], labels.loc[idx]


def build_set(all_files, directory, load_sample, markers, n=N_CELLS_PER_SAMPLE):
  """
  Crea los tensores de PyTorch a partir de una lista de muestras de citometría de flujo.

  Args:
    all_files (pd.DataFrame): DataFrame que contiene los identificadores de los ficheros
      a procesar
    directory (Path): Ruta base del dataset que contiene las carpetas
    target_markers (list[str], optional): Lista de marcadores a extraer como variables de
      entrada X
    target_label (list[str], optional): Lista con el nombre de la columna objetivo a predecir
    n (int, optional): Número de células a submuestrear por cada muestra

    Returns:
      tuple[torch.Tensor, torch.Tensor]:
        - X_tensor (torch.Tensor): (N_total_células, len(target_markers)) con los valores de dispersión y fluorescencia.
        - y_tensor (torch.Tensor): (N_total_células, 1) con las etiquetas binarias (0 = no blasto, 1 = blasto).
    """
  filenames = all_files.iloc[:, 0]
  
  x_features = []
  y_labels = []

  print("cambio de set")
  for row in filenames:
    print(row)
    try:
      data, labels = load_sample(directory, row, markers)
    except KeyError as e:
      print(f"Muestra {row} omitida: falta el marcador {e}")
      continue
    if len(data.index) == 0:
      print(f"Muestra {row} omitida: falta de células")
      continue
    k = min(n, len(data.index))
    idx = data.sample(k, axis=0).index
    x_features.append(data.loc[idx])
    y_labels.append(labels.loc[idx])

  X = pd.concat(x_features, ignore_index=True).values.astype(np.float32)
  y = pd.concat(y_labels, ignore_index=True).values.astype(np.uint8)
  return X, y

def _create(path, load, markers, tube=None): 
  train_ids, validation_ids, test_ids = split_from(path, 0.2, 0.2, tube) 
  x_train, y_train = build_set(train_ids, path, load, markers)
  x_validation, y_validation = build_set(validation_ids, path, load, markers)
  x_test, y_test = build_set(test_ids, path, load, markers)
  
  directory = Path("data")
  if tube is not None:
    directorio = directory / tube
  os.makedirs(path / directory, exist_ok=True)
  np.savez_compressed(path / directory / "train.npz", X=x_train, y=y_train)
  np.savez_compressed(path / directory / "validation.npz", X=x_validation, y=y_validation)
  np.savez_compressed(path / directory / "test.npz",  X=x_test,  y=y_test)

def create_model_BLAST():
  """
  Crea los conjuntos de datos de entrenamiento, validación y test y los guarda como tensores
  en el directorio data, si no existe lo crea. 

  Args:
    database (str): Nombre del directorio correspondiente a la base de datos
  """
  path = Path('.')
  db_path = path / 'BLAST110'
  _create(db_path, load_blast_sample, MARKERS_BLAST)

def create_model_LAIP():
  path = Path('.')
  db_path = path / 'LAIP29'
  _create(db_path, load_laip_sample, MARKERS_TUBE1, "P1")
  _create(db_path, load_laip_sample, MARKERS_TUBE2, "P2")
  _create(db_path, load_laip_sample, MARKERS_TUBE3, "P3")
  _create(db_path, load_laip_sample, MARKERS_TUBE4, "P4")

create_model_BLAST()
create_model_LAIP()
