import numpy as np
import gensim.downloader as api

# Cargar GloVe 50D
glove = api.load('glove-wiki-gigaword-50')

# Polos (Bolukbasi et al., 2016)
female = ['she', 'woman', 'female', 'daughter', 'mother', 'girl']
male   = ['he', 'man',   'male',   'son',      'father', 'boy']

def get_semaxis(list1, list2, model, embedding_size, normalize=True):
    """Calcula el embedding de un eje semántico dadas dos listas de palabras polo."""
    # 1) Filtrar palabras fuera de vocabulario (OOV)
    in_vocab1 = [w for w in list1 if w in model.key_to_index]
    in_vocab2 = [w for w in list2 if w in model.key_to_index]
    if not in_vocab1 or not in_vocab2:
        raise ValueError("Alguna de las listas quedó vacía tras filtrar palabras OOV.")

    # 2) Obtener embeddings y promediar cada polo
    v_plus  = np.vstack([model[w] for w in in_vocab1])   # polo positivo
    v_minus = np.vstack([model[w] for w in in_vocab2])   # polo negativo
    v_plus_mean  = v_plus.mean(axis=0)
    v_minus_mean = v_minus.mean(axis=0)

    # 3) Diferencia de medios = eje semántico
    sem_axis = v_plus_mean - v_minus_mean

    # (Opcional) normalizar a unidad, útil para proyecciones consistentes
    if normalize:
        norm = np.linalg.norm(sem_axis)
        if norm > 0:
            sem_axis = sem_axis / norm

    # Sanity check
    assert sem_axis.size == embedding_size
    return sem_axis

# Calcular el eje de género
gender_axis = get_semaxis(
    list1=female,
    list2=male,
    model=glove,
    embedding_size=50
)

gender_axis, gender_axis.shape  # -> (array([...]), (50,))

def projection_score(word, axis, model):
    return np.dot(model[word], axis)

print("nurse:", projection_score("nurse", gender_axis, glove))
print("engineer:", projection_score("engineer", gender_axis, glove))
print("doctor:", projection_score("doctor", gender_axis, glove))

