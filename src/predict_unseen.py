import os
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# --- Settings ---
UNSEEN_DIR = "data/unseen/"
MODEL_PATH = (
    "models/cancer_classifier_20250428_1420.keras"  # Update to your latest model
)
IMG_SIZE = (96, 96)

# --- Load model ---
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")


# --- Preprocess function ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load {img_path}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# --- Predict on unseen images ---
results = []

true_labels = []
predicted_labels = []

for label_folder, true_label in [("benign", 0), ("malignant", 1)]:
    folder_path = os.path.join(UNSEEN_DIR, label_folder)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder {folder_path} does not exist, skipping...")
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            processed_img = preprocess_image(img_path)
            if processed_img is None:
                continue
            prediction = model.predict(processed_img, verbose=0)
            predicted_class = int(prediction[0][0] > 0.5)  # 0 or 1
            confidence = float(prediction[0][0])

            true_labels.append(true_label)
            predicted_labels.append(predicted_class)

            results.append(
                {
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": predicted_class,
                    "confidence": confidence,
                }
            )

            print(
                f"{label_folder}/{filename}: Predicted {predicted_class} (confidence {confidence:.4f})"
            )

# --- Evaluation ---
print("\n📊 Classification Report:")
print(
    classification_report(
        true_labels, predicted_labels, target_names=["Non-cancerous", "Cancerous"]
    )
)

# --- Save outputs ---
os.makedirs("outputs", exist_ok=True)

# Save CSV
results_df = pd.DataFrame(results)
csv_path = "outputs/unseen_predictions.csv"
results_df.to_csv(csv_path, index=False)
print(f"📄 Detailed results saved to {csv_path}")

# Save coloured Excel
excel_path = "outputs/unseen_predictions_styled.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    results_df.to_excel(writer, index=False, sheet_name="Predictions")
    workbook = writer.book
    worksheet = writer.sheets["Predictions"]

    # Define formats
    correct_format = workbook.add_format(
        {"bg_color": "#C6EFCE", "font_color": "#006100"}
    )  # Green
    incorrect_format = workbook.add_format(
        {"bg_color": "#FFC7CE", "font_color": "#9C0006"}
    )  # Red

    # Apply conditional formatting
    worksheet.conditional_format(
        "A2:D{}".format(len(results_df) + 1),
        {"type": "formula", "criteria": "=$B2=$C2", "format": correct_format},
    )
    worksheet.conditional_format(
        "A2:D{}".format(len(results_df) + 1),
        {"type": "formula", "criteria": "=$B2<>$C2", "format": incorrect_format},
    )

print(f"📄 Styled Excel sheet saved to {excel_path}")

# --- Confusion matrix ---
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-cancerous", "Cancerous"],
    yticklabels=["Non-cancerous", "Cancerous"],
)
plt.title("Confusion Matrix on Unseen Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = "outputs/confusion_matrix_unseen.png"
plt.savefig(cm_path)
plt.show()

print(f"📷 Confusion matrix saved to {cm_path}")
