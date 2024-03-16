import os
import random
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

circle = 'shapes-dataset/train/Circle'
triangle = 'shapes-dataset/train/Triangle'
square = 'shapes-dataset/train/Square'
star = 'shapes-dataset/train/Star'
all_training = [circle, triangle, star, square]
circle_t = 'shapes-dataset/test/Circle'
triangle_t = 'shapes-dataset/test/Triangle'
square_t = 'shapes-dataset/test/Square'
star_t = 'shapes-dataset/test/Star'
all_testing = [circle_t, triangle_t, star_t, square_t]
labels = ['circle', 'triangle', 'star', 'square']


class ShapeRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(ShapeRecognitionModel, self).__init__()
        self.resnet18 = models.resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


official_model = ShapeRecognitionModel()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY']
socketio = SocketIO(app)

database = {'answers': [], 'currentAnswer': 0, 'difficulty': 0}


@socketio.on('connect')
def handle_connect(message):
    print("You are connected")


@socketio.on('message')
def handle_message(message):
    if message:
        print('there is a message ' + str(message['difficulty']))
        database['answers'] = []
        if message['difficulty'] == 1:
            database["difficulty"] = 6
        elif message['difficulty'] == 2:
            database["difficulty"] = 9
        else:
            database["difficulty"] = 0
            database["difficulty"] = 4
    i = 0
    total_shapes = database['difficulty']
    while i < total_shapes:
        database['answers'].append(random.randint(0, 3))
        i += 1
    database['currentAnswer'] = 0
    emit('response', {"answers": database['answers'], "currentAnswer": database['currentAnswer']})


@socketio.on('stream')
def handle_stream(data):
    py_arr = np.frombuffer(data['blob'], np.uint8)
    loaded_model = ShapeRecognitionModel()
    loaded_model.load_state_dict(torch.load('model_weights.pth'))
    loaded_model.eval()
    decoded_image = cv2.imdecode(py_arr, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(decoded_image, (48, 48))
    gray_tensor = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0) / 255.0
    gray_tensor = gray_tensor.permute(0, 3, 1, 2)
    logits = loaded_model.forward(gray_tensor)
    probabilities = torch.softmax(logits, dim=1)
    prediction = 0.0
    guess = 0
    for index, num in enumerate(probabilities[0]):
        print(probabilities[0])
        if num > prediction:
            guess = index
            prediction = num
    print("Sensed " + labels[guess])
    print("Should be " + labels[database['answers'][database['currentAnswer']]])
    current_session = Session.query.first()
    current_user = User.query.filter_by(username=current_session.username).first()
    if database['answers'][database['currentAnswer']] == guess:
        database['currentAnswer'] += 1
        current_user.exp += 100
        current_user.lifetimeA += 1
        current_user.lifetimeB += 1
        if current_user.exp == 300:
            current_user.rank = "Sharp"
        db.session.commit()
        emit('response', {"success": "correct", "currentAnswer": database['currentAnswer']})
    else:
        current_user.exp -= 100
        current_user.lifetimeB += 1
        db.session.commit()
        database['currentAnswer'] += 1
        emit('response', {"error": "Try Again", "currentAnswer": database['currentAnswer']})


def prepare_data(inputs, validate):
    print('Preparing dataset...')
    data = []
    validation_set = []
    inputs_val = inputs
    labels_to_use = []
    labels_val = []

    for index, folder in enumerate(inputs_val):
        for i, filename in enumerate(os.listdir(folder)):
            file_path = os.path.join(folder, filename)

            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                gray_image = cv2.resize(gray_image, (48, 48))
                if i % 5 == 0:
                    validation_set.append(gray_image)
                    labels_val.append(index)
                else:
                    data.append(gray_image)
                    labels_to_use.append(index)

    labels_tensors = torch.tensor(labels_to_use, dtype=torch.long)
    labels_val_tensors = torch.tensor(labels_val, dtype=torch.long)
    validation_set_np = np.array(validation_set).astype(np.float32) / 255.0
    validation_set_tensors = torch.tensor(validation_set_np, dtype=torch.float32)
    validation_set_tensors = validation_set_tensors.permute(0, 3, 1, 2)
    validation_data = TensorDataset(validation_set_tensors, labels_val_tensors)
    val_loader = DataLoader(validation_data, batch_size=32, shuffle=True)
    data_np = np.array(data).astype(np.float32) / 255.0
    data_tensors = torch.tensor(data_np, dtype=torch.float32)
    data_tensors = data_tensors.permute(0, 3, 1, 2)
    dataset = TensorDataset(data_tensors, labels_tensors)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    if validate:
        return dataloader, val_loader
    else:
        return dataloader


def test_model(validation_set):
    print('Testing model...')
    correct = 0
    total = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    if validation_set:
        loaded_model = official_model
        testing_data = validation_set
    else:
        testing_data = prepare_data(all_testing, False)
        loaded_model = ShapeRecognitionModel()
        loaded_model.load_state_dict(torch.load('model_weights.pth'))
    loaded_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testing_data:
            outputs = loaded_model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(probabilities, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(testing_data)
    avg_accuracy = correct / total
    print(str(correct) + " Correct " + "Out of " + str(total))
    return avg_loss, avg_accuracy, all_labels, all_preds


def plot_scatterplot(epochs, losses):
    plt.scatter(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.show()


def plot_histogram(all_labels, all_preds):
    shapes = {
        'circle': {'correct': 0, 'incorrect': 0},
        'triangle': {'correct': 0, 'incorrect': 0},
        'star': {'correct': 0, 'incorrect': 0},
        'square': {'correct': 0, 'incorrect': 0},
    }

    for pred, label in zip(all_preds, all_labels):
        shape_name = labels[label]
        if pred == label:
            shapes[shape_name]['correct'] += 1
        else:
            shapes[shape_name]['incorrect'] += 1

    correct_counts = [shapes[shape]['correct'] for shape in shapes]
    incorrect_counts = [shapes[shape]['incorrect'] for shape in shapes]
    shape_names = list(shapes.keys())
    bar_width = 0.35
    index = np.arange(len(shapes))
    plt.bar(index, correct_counts, bar_width, label='Correct')
    plt.bar(index + bar_width, incorrect_counts, bar_width, label='Incorrect')
    plt.xlabel('Shapes')
    plt.ylabel('Number of Predictions')
    plt.title('Histogram of Prediction Correctness per Shape')
    plt.xticks(index + bar_width / 2, shape_names)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model():
    print('Training model...')
    writer = SummaryWriter('/tmp/tensorboard')
    dataloader, val_loader, = prepare_data(all_training, True)
    optimizer = SGD(official_model.parameters(), lr=0.001, momentum=0.8, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    official_model.train()
    score = 0.0
    epoch_losses = []
    all_epoch_preds = []
    all_epoch_labels = []
    for epoch in range(9):
        epoch_loss = []
        epoch_preds = []
        epoch_labels = []
        all_labels = None
        all_preds = None
        print("epoch " + str(epoch + 1))
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx == 0:
                optimizer.zero_grad()
                outputs = official_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
                epoch_loss.append(loss.item())
                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
                writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
        avg_loss, avg_accuracy, all_labels, all_preds = test_model(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', avg_accuracy, epoch)
        if avg_accuracy > score:
            score = avg_accuracy
            torch.save(official_model.state_dict(), 'model_weights.pth')
        avg_loss = np.mean(epoch_loss)
        epoch_losses.append(avg_loss)
        all_epoch_preds.extend(epoch_preds)
        all_epoch_labels.extend(epoch_labels)
    plot_scatterplot(range(1, 10), epoch_losses)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    plot_histogram(all_labels, all_preds)


def test_photo(inputs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    for index, folder in enumerate(inputs):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                gray_image = cv2.resize(gray_image, (48, 48))
                cv2.imshow('Median Blurred Image', gray_image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()


basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'memorinc.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False)
    password = db.Column(db.String(30), nullable=False)
    lifetimeA = db.Column(db.Integer, nullable=False, default=0)
    lifetimeB = db.Column(db.Integer, nullable=False, default=0)
    rank = db.Column(db.String(10), nullable=False, default="All There")
    exp = db.Column(db.Integer, nullable=False, default=0)

    def __repr__(self):
        return f'<User {self.username}>'


@app.route('/play')
def hello_world():
    current_session = Session.query.first()
    if current_session is None:
        return render_template("login.html")
    current_user_ = User.query.filter_by(username=current_session.username).first()
    return render_template("home.html", username=current_user_.username, lifetimeA=current_user_.lifetimeA,
                           lifetimeB=current_user_.lifetimeB, rank=current_user_.rank, exp=current_user_.exp)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user is None:
            new_user = User(username=username, password=password, lifetimeA=0, lifetimeB=0, rank="All There", exp=0)
            db.session.add(new_user)
            db.session.commit()
            new_session = Session(username=username)
            db.session.add(new_session)
            db.session.commit()
            return render_template("landing.html", flash="User Created!")
        else:
            if user.password == password:
                new_session = Session(username=username)
                db.session.add(new_session)
                db.session.commit()
                return render_template("landing.html", flash="Login Successful!")
            else:
                return render_template("login.html", flash="Invalid username or password!")


class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    logged_in = db.Column(db.Boolean, default=True, nullable=False)
    username = db.Column(db.String(100), db.ForeignKey('user'))


def test():
    db.create_all()
    db.session.query(Session).delete()
    db.session.commit()


@app.route('/')
def home():
    current_session = Session.query.first()
    if current_session is None:
        return render_template("landing.html", session=False)
    else:
        return render_template("landing.html", session=True)


@app.route('/signout')
def signout():
    db.session.query(Session).delete()
    db.session.commit()
    return render_template("landing.html", flash="Logged Out", session=False)

if not os.path.exists('model_weights.pth'):
    train_model()
else:
    test_model(None)
with app.app_context():
    test()
if __name__ == '__main__':
    socketio.run(app)
# flask run -h localhost -p 8080
