from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

# Try loading with trust_remote_code first
try:
    metric = evaluate.load('accuracy', trust_remote_code=True)
except:
    # Fallback to sklearn if evaluate doesn't work
    from sklearn.metrics import accuracy_score
    metric = None

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    if metric is not None:
        return metric.compute(predictions=predictions, references=labels)
    else:
        # Use sklearn as fallback
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

def get_class_weights(df):
    # Get unique classes and convert to numpy array
    unique_classes = sorted(df['label'].unique().tolist())
    classes_array = np.array(unique_classes)
    
    class_weights = compute_class_weight("balanced",
                         classes=classes_array,  # Now it's a numpy array
                         y=df['label'].tolist()
                         )
    return class_weights