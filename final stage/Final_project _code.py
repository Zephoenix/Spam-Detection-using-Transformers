# Import necessary libraries for data processing, model training, and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score, 
    recall_score, roc_auc_score, roc_curve, classification_report, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Install and import HuggingFace libraries for transformers and datasets
%pip install transformers datasets torch transformers[torch]
from transformers import BertTokenizerFast
import torch
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
import json

# Load email dataset and handle missing values in 'subject' and 'message'
email = pd.read_csv("emails.csv")
email['subject'] = email['subject'].fillna('')
email['message'] = email['message'].fillna('')

# Combine 'subject' and 'message' into a single 'text' column
email = email[["class", "subject", "message"]]
email['text'] = email['subject'] + " " + email['message']
email.drop(columns=['subject', 'message'], inplace=True)

# Convert 'class' column to binary (1 for spam, 0 for ham)
email['class'] = email['class'].apply(lambda x: 1 if x == 'spam' else 0)

# Split dataset into training and testing sets
X = email['text']
y = email['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize training and testing text data with truncation and padding
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, 
                            max_length=32)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, 
                           max_length=32)

# Define a custom dataset class to handle tokenized inputs and labels
class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create dataset objects for training and testing
train_dataset = SpamDataset(train_encodings, y_train.tolist())
test_dataset = SpamDataset(test_encodings, y_test.tolist())

# Load pre-trained BERT model for sequence classification with 2 output labels (spam/ham)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set training arguments for HuggingFace Trainer API
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save model checkpoints and logs
    eval_strategy="epoch",   # Evaluate the model at the end of each epoch
    num_train_epochs=2,      # Train for 2 epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=250,        # Number of warmup steps for learning rate
    weight_decay=0.01,       # L2 regularization to prevent overfitting
    logging_dir='./logs',    # Directory for logging
    logging_steps=10,        # Log metrics every 10 steps
    save_strategy="epoch",   # Save checkpoints at the end of each epoch
    load_best_model_at_end=True,  # Load the best model after training
    gradient_accumulation_steps=4,  # Gradient accumulation for larger effective batch size
    fp16=True               # Enable mixed-precision training for efficiency
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model on the test set
metrics = trainer.evaluate()

# Generate predictions on the test dataset
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

# Convert logits to class predictions (argmax)
predicted_classes = np.argmax(logits, axis=1)

# Compute evaluation metrics
accuracy = accuracy_score(labels, predicted_classes)
precision = precision_score(labels, predicted_classes)
recall = recall_score(labels, predicted_classes)
f1 = f1_score(labels, predicted_classes)

# Print classification report
print(pd.DataFrame(classification_report(predicted_classes, labels, 
                                         output_dict=True, target_names=['ham', 'spam'])))

# Plot confusion matrix
cm_test = confusion_matrix(predicted_classes, labels)
plt.figure(figsize=(6, 4))
print("\nConfusion Matrix:")
sns.heatmap(cm_test, annot=True, xticklabels=['ham', 'spam'], 
            yticklabels=['ham', 'spam'], cmap="Oranges", fmt='.4g')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Train set ROC curve
predictions_train = trainer.predict(train_dataset)
logits_train = predictions_train.predictions
y_train = predictions_train.label_ids
y_pred_prob_train = F.softmax(torch.tensor(logits_train), dim=1).numpy()[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_prob_train)
roc_auc_train = auc(fpr_train, tpr_train)

# Test set ROC curve
y_pred_prob_test = F.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
fpr_test, tpr_test, _ = roc_curve(labels, y_pred_prob_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves
plt.figure()
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot(fpr_train, tpr_train, color='red', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the trained model and tokenizer
model.save_pretrained("./saved_bert_model")
tokenizer.save_pretrained("./saved_bert_model")

# Initialize lists to store metrics
epoch = []
loss = []

# Load the log file
log_file_path = "./results/checkpoint-420/trainer_state.json"
with open(log_file_path, "r") as file:
    logs = json.load(file)

# Iterate over the log history and extract metrics
for log in logs['log_history'][:-1]:
    epoch_value = log.get('epoch', None)
    loss_value = log.get('loss', None)

    # Append rounded values (if they exist)
    if epoch_value is not None:
        epoch.append(round(epoch_value, 4))
    if loss_value is not None:
        loss.append(round(loss_value, 4))

loss.append(round(logs['log_history'][-1].get('eval_loss'), 4))
eval_loss = [0.0573, 0.0398]  # Evaluation losses at end of epochs

# Plot of epoch vs training loss
plt.figure(figsize=(10, 6))
plt.plot(epoch, loss, label='Training Loss', marker='o')
plt.axhline(eval_loss[-1], color='red', linestyle='--', label='Final Eval Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# Load personal email dataset
new_emails = pd.read_csv('personal_emails.csv')
new_emails['Subject'] = new_emails['Subject'].fillna('')
new_emails['Body'] = new_emails['Body'].fillna('')
new_emails['text'] = new_emails['Subject'] + " " + new_emails['Body']
new_emails.drop(columns= ['Subject', 'Body'], inplace= True)
new_emails['class'] = new_emails['Classification'].apply(lambda x: 1 if x == 'Spam' else 0)
new_emails.drop(columns = ['Classification'], inplace = True)
new_encodings = tokenizer(list(new_emails['text']), truncation= True, padding= True, max_length= 32)

# Define a custom PyTorch dataset for new data
class NewDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings  # Encoded input data (tokenized emails)

    def __len__(self):
        return len(self.encodings['input_ids'])  # Length of the dataset
    
    def __getitem__(self, idx):
        # Retrieve individual tokenized email data at a given index
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Use the custom dataset to predict on new emails
new_emails_data = NewDataset(new_encodings)  # Create dataset object

# Generate predictions using the trained model
predictions = trainer.predict(new_emails_data)

# Extract logits (raw model outputs) and convert to class predictions
logits = predictions.predictions
predicted_classes = torch.argmax(torch.tensor(logits), dim=1).numpy()  # Take argmax to get class

# Add predictions back to the new_emails DataFrame
new_emails['predicted_class'] = predicted_classes
new_emails['predicted_label'] = new_emails['predicted_class'].apply(lambda x: 'spam' if x == 1 else 'ham')

# Evaluate the model's performance (accuracy, precision, recall, F1-score)
accuracy = accuracy_score(new_emails['class'], new_emails['predicted_class'])
precision = precision_score(new_emails['class'], new_emails['predicted_class'])
recall = recall_score(new_emails['class'], new_emails['predicted_class'])
f1 = f1_score(new_emails['class'], new_emails['predicted_class'])

# Load the ChatGPT dataset for predictions
new_emails = pd.read_csv('LLMEmails.csv')  # Load new emails

# Fill missing values in 'Subject' and 'Body' columns
new_emails['Subject'] = new_emails['Subject'].fillna('')
new_emails['Body'] = new_emails['Body'].fillna('')

# Combine 'Subject' and 'Body' into a single 'text' column for tokenization
new_emails['text'] = new_emails['Subject'] + " " + new_emails['Body']
new_emails.drop(columns=['Subject', 'Body'], inplace=True)  # Drop unnecessary columns

# Convert class labels to binary (1 for Spam, 0 for Ham)
new_emails['class'] = new_emails['Label'].apply(lambda x: 1 if x == 'Spam' else 0)
new_emails.drop(columns=['Label'], inplace=True)  # Drop original label column

# Tokenize the new email text data
new_encodings = tokenizer(
    list(new_emails['text']),
    truncation=True,
    padding=True,
    max_length=32
)

# Create a dataset object using the custom NewDataset class
new_emails_data = NewDataset(new_encodings)

# Generate predictions using the trained model
predictions = trainer.predict(new_emails_data)

# Extract logits and convert to class predictions
logits = predictions.predictions
predicted_classes = torch.argmax(torch.tensor(logits), dim=1).numpy()

# Add predictions back to the DataFrame
new_emails['predicted_class'] = predicted_classes
new_emails['predicted_label'] = new_emails['predicted_class'].apply(lambda x: 'spam' if x == 1 else 'ham')

# Evaluate model performance using accuracy, precision, recall, and F1 score
accuracy = accuracy_score(new_emails['class'], new_emails['predicted_class'])
precision = precision_score(new_emails['class'], new_emails['predicted_class'])
recall = recall_score(new_emails['class'], new_emails['predicted_class'])
f1 = f1_score(new_emails['class'], new_emails['predicted_class'])

