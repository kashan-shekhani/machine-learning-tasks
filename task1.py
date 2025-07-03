from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

emails = [
    "Get rich quick! Click here to win",
    "Your Flight Reservation Confirmation",
    "You've won a lottery! Click here to claim your prize",
    "Reminder: Submission Deadline for Research Paper",
    "Urgent: Your Account Has Been Compromised",
    "Review this pdf send by the company",
]

labels = ["1", "0", "1", "0", "1", "0"]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0 )

user_email = input("Enter the email text to classify: ")

print("Accuracy:", accuracy)
print("Report\n", report)

new_email = [user_email]
new_email_vector = vectorizer.transform(new_email)
predicted_label = model.predict(new_email_vector)

if predicted_label[0] == "0":
    print("This is not a spam email\n")
else:
    print("This is spam email\n")