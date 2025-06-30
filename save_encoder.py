import pickle
from sklearn.preprocessing import LabelEncoder

# Enter the same list of labels used in your model
labels = ['label1', 'label2', 'label3']  # <- replace with actual classes

encoder = LabelEncoder()
encoder.fit(labels)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Encoder saved as label_encoder.pkl")
