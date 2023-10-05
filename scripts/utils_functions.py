import numpy as np

subject_ids = list(range(10, 43))
wrong_subjects = [11, 24, 26]
valid_subjects = np.array(list(set(subject_ids) - set(wrong_subjects)))

def load_scores(dir, file, average):
    """
    Loads decoding scores for all valid subjects. The valid subjects list is defined by 
    subtracting discarded subjects from the whole sample. If average == True, it returns
    an array with the grand mean (averaged data for all subjects). If average == False,
    it returns an array with data for all subjects, not averaged.
    Loaded data must be in .npy format. 

    Args: 
        dir (str): directory where the data files are stored
        file (str): file name
        average (bool): whether we want the scores averaged for all subjects or not

    Returns:
        scores (array): array with the loaded scores
    """  
    scores = []
    
    for _, id in enumerate(valid_subjects):
        subject_scores = np.load(f'{dir}/S{id}_{file}.npy')
        if np.any(scores):
            scores = np.row_stack((scores, subject_scores))
        else:
            scores = subject_scores

    if average == True:
        scores = np.mean(scores, axis = 0)
    else:
        pass
    
    return scores