from model import train_and_evaluate
def test_model_accuracy():
    accuracy = train_and_evaluate()
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy}"
