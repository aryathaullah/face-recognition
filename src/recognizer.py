from .metrics import euclidean_distance, cosine_similarity

def recognize(test_vector, X_train, y_train, method="euclidean"):
    best_score = None
    best_label = None

    for x, label in zip(X_train, y_train):
        if method == "euclidean":
            score = euclidean_distance(test_vector, x)
            is_better = best_score is None or score < best_score
        else:
            score = cosine_similarity(test_vector, x)
            is_better = best_score is None or score > best_score

        if is_better:
            best_score = score
            best_label = label

    return best_label, best_score
