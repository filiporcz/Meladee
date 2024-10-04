import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QFrame, QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect, QSlider
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QUrl, QTime
from PyQt5.QtGui import QPixmap, QIcon, QFont, QFontDatabase, QColor, QMovie
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pyqtgraph as pg
from pydub import AudioSegment
import numpy as np
from scipy import stats
import pandas as pd
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model

model = load_model('E:/Meladee/CNN model.keras')

def create_bold_font(size):
    font_id = QFontDatabase.addApplicationFont('E:/Meladee/assets/fonts/Baloo2-VariableFont_wght.ttf')
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
    font = QFont(font_family, size)
    font.setWeight(99)
    return font

class IconButton(QPushButton):
    def __init__(self, icon_paths, initial_icon_key, parent=None):
        super().__init__(parent)
        self.icon_paths = icon_paths
        self.icon_key = initial_icon_key
        self.update_hover_icon_key()
        self.set_icon(self.icon_key)
        self.setFlat(True)
        self.setStyleSheet("background-color: transparent;")

    def set_icon(self, icon_key):
        """Set the button's icon based on the provided icon key."""
        self.setIcon(QIcon(self.icon_paths[icon_key]))
        self.current_icon = icon_key

    def update_icon_keys(self, new_icon_key):
        """Update the icon keys based on the application's state changes."""
        self.icon_key = new_icon_key
        self.update_hover_icon_key()
        self.set_icon(self.icon_key)

    def update_hover_icon_key(self):
        """Update the hover icon key based on the current icon key."""
        self.hover_icon_key = f"{self.icon_key}_pressed" if f"{self.icon_key}_pressed" in self.icon_paths else self.icon_key

    def enterEvent(self, event):
        """Change the icon to the hovered version when the mouse enters the button area."""
        super().enterEvent(event)
        self.set_icon(self.hover_icon_key)

    def leaveEvent(self, event):
        """Revert to the original icon when the mouse leaves the button area."""
        super().leaveEvent(event)
        self.set_icon(self.icon_key)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meladee - Music Genre Classification Tool")
        self.setFixedSize(QSize(1200, 800))

        self.setup_initial_view()
    
    def setup_initial_view(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setupUI(central_widget)

    def setupUI(self, central_widget):
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 50, 10, 150)

        logo = QLabel()
        pixmap = QPixmap('E:/Meladee/assets/logo.png')
        scaled_pixmap = pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(scaled_pixmap)
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)
        layout.addSpacing(-10)

        font_id = QFontDatabase.addApplicationFont('E:/Meladee/assets/fonts/Baloo2-VariableFont_wght.ttf')
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]

        title = QLabel("Meladee")
        title.setFont(create_bold_font(32))
        title.setStyleSheet("color: #3A0073;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(-15)

        subtitle = QLabel("Music Genre Classification Tool")
        subtitle.setFont(create_bold_font(15))
        subtitle.setStyleSheet("color: #3A0073;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        self.error_message_layout = QHBoxLayout()
        self.error_message_layout.setAlignment(Qt.AlignCenter)
        self.error_message_layout.setSpacing(10)

        self.error_icon_label = QLabel()
        error_pixmap = QPixmap('E:/Meladee/assets/error.svg').scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.error_icon_label.setPixmap(error_pixmap)
        self.error_icon_label.setStyleSheet("border: none;")

        self.error_message_label = QLabel("Unsupported file format, please try again")
        self.error_message_label.setFont(create_bold_font(13))
        self.error_message_label.setStyleSheet("""
            color: #FF2020; 
            background-color: transparent;
            border: none;
            padding: 0;
            margin: 0;
        """)

        self.error_message_box = QWidget()
        self.error_message_box.setLayout(self.error_message_layout)
        self.error_message_box.setStyleSheet("""
            background-color: transparent;
            border-radius: 10px;
            border: 3px solid #FF2020;
            padding: 0;
            margin: 0;
        """)

        self.error_message_layout.addWidget(self.error_icon_label)
        self.error_message_layout.addWidget(self.error_message_label)

        self.error_message_box.setFixedSize(400, 50)
        self.error_message_box.hide()
        layout.addWidget(self.error_message_box, alignment=Qt.AlignCenter)

        self.subtitle_to_file_upload_spacer_widget = QWidget()
        self.subtitle_to_file_upload_spacer_widget.setFixedHeight(60)
        layout.addWidget(self.subtitle_to_file_upload_spacer_widget)
        
        self.file_upload_layout = QHBoxLayout()
        self.file_upload_layout.addStretch()
        self.file_upload_label = QLabel("File Upload")
        self.file_upload_label.setFont(create_bold_font(14))
        self.file_upload_label.setStyleSheet("color: #3A0073;")
        self.file_upload_layout.addWidget(self.file_upload_label)

        self.file_upload_layout.addSpacing(100)

        self.max_file_size_label = QLabel("50MB Maximum File Size")
        self.max_file_size_label.setFont(create_bold_font(12))
        self.max_file_size_label.setStyleSheet("color: #3A0073;")
        self.file_upload_layout.addWidget(self.max_file_size_label)

        self.file_upload_layout.addStretch()
        layout.addLayout(self.file_upload_layout)

        layout.addSpacing(-5)

        self.drop_area = DropArea(self)
        self.drop_area.fileDropped.connect(self.switch_to_result_view)
        self.drop_area.setAcceptDrops(True)
        self.drop_area.setFrameShape(QFrame.StyledPanel)
        self.drop_area.setFixedSize(QSize(400, 320))
        self.drop_area.setStyleSheet("""
            QFrame {
                background-color: white;
                border-style: dashed;
                border-radius: 15px;
                border-color: #3A0073;
                border-width: 2px;
            }""")
        drop_area_layout = QVBoxLayout(self.drop_area)
        drop_area_layout.setAlignment(Qt.AlignCenter)

        self.drop_area.formatError.connect(self.show_error_message)
        
        picture_label = QLabel(self.drop_area)
        picture_pixmap = QPixmap('E:/Meladee/assets/cloud-upload-signal.svg')
        picture_label.setPixmap(picture_pixmap.scaled(60, 60, Qt.KeepAspectRatio))
        picture_label.setStyleSheet("border: none;")
        drop_area_layout.addWidget(picture_label, alignment=Qt.AlignHCenter)

        drag_drop_label = QLabel("Drag and Drop File Here", self.drop_area)
        drag_drop_label.setFont(create_bold_font(15))
        drag_drop_label.setStyleSheet("border: none; color: #3A0073;")
        drop_area_layout.addWidget(drag_drop_label, alignment=Qt.AlignHCenter)

        or_label = QLabel("OR", self.drop_area)
        or_label.setFont(create_bold_font(15))
        or_label.setStyleSheet("border: none; color: #3A0073;")
        drop_area_layout.addWidget(or_label, alignment=Qt.AlignHCenter)

        btn_upload = QPushButton("Browse Files")
        btn_upload.setFont(create_bold_font(15))
        btn_upload.setStyleSheet("""
            QPushButton {
                background-color: #3A0073;
                color: white;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #710CCF;

            }
        """)
        btn_upload.setFixedHeight(50)
        self.apply_shadow(btn_upload)
        drop_area_layout.addWidget(btn_upload)
        drop_area_layout.setSpacing(15)
        btn_upload.clicked.connect(self.browse_files)

        spacer = QSpacerItem(10, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)
        drop_area_layout.addSpacerItem(spacer)

        centering_layout = QHBoxLayout()
        centering_layout.addWidget(self.drop_area, 0, Qt.AlignCenter)
        layout.addLayout(centering_layout)

        self.accepted_formats_label = QLabel("Accepted Formats: .mp3 .wav .flac .aiff")
        self.accepted_formats_label.setFont(create_bold_font(12))
        self.accepted_formats_label.setStyleSheet("color: #3A0073;")
        layout.addWidget(self.accepted_formats_label, alignment=Qt.AlignCenter)
        layout.addSpacing(-100)

    def show_initial_view(self):
        self.setup_initial_view()

    def apply_shadow(self, widget, color='#000000', offset=(0, 0), blur_radius=15, enabled=True):
        shadow = QGraphicsDropShadowEffect()
        shadow.setColor(QColor(color))
        shadow.setOffset(*offset)
        shadow.setBlurRadius(blur_radius)
        shadow.setEnabled(enabled)
        widget.setGraphicsEffect(shadow)

    def browse_files(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.mp3 *.wav *.flac *.aiff)")
        if file_name:
            print("Selected file:", file_name)
            audio, sr = librosa.load(file_name, sr=None)
            self.switch_to_result_view(file_path=file_name, sr=sr)
        
    def show_error_message(self):
        self.error_message_box.show()
        self.subtitle_to_file_upload_spacer_widget.hide()
    
    def hide_error_message(self):
        self.error_message_box.hide()
        self.subtitle_to_file_upload_spacer_widget.show()

    def switch_to_result_view(self, file_path='', sr=None):
        predicted_genre_label = ResultWindow.predict_genre(file_path, sr)
        self.result_window = ResultWindow(file_path=file_path, sr=sr, predicted_genre=predicted_genre_label)
        self.result_window.backRequested.connect(self.show_initial_view)
        self.setCentralWidget(self.result_window)
    
class DropArea(QFrame):
    fileDropped = pyqtSignal(str)
    formatError = pyqtSignal()

    def __init__(self, parent=None):
        super(DropArea, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-style: dashed;
                border-radius: 15px;
                border-color: #3A0073;
                border-width: 2px;
            }""")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame {
                    background-color: #F5EBFF;
                    border-style: solid;
                    border-radius: 15px;
                    border-color: #3A0073;
                    border-width: 2px;
                }""")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-style: dashed;
                border-radius: 15px;
                border-color: #3A0073;
                border-width: 2px;
            }""")

    def dropEvent(self, event):
        url = event.mimeData().urls()[0]
        file_path = url.toLocalFile()
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.mp3', '.wav', '.flac', '.aiff']:
            event.acceptProposedAction()
            self.fileDropped.emit(file_path)
        else:
            self.formatError.emit()
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-style: dashed;
                border-radius: 15px;
                border-color: #3A0073;
                border-width: 2px;
            }""")

class ResultWindow(QWidget):
    backRequested = pyqtSignal()

    def __init__(self, file_path, sr, predicted_genre, parent=None):
        super(ResultWindow, self).__init__(parent)
        self.file_path = file_path
        self.sr = sr
        self.predicted_genre = predicted_genre
        predicted_genre_label = ResultWindow.predict_genre(file_path, sr)
        self.file_path = file_path
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.LowLatency)
        self.model = load_model('E:/Meladee/CNN model.keras')

        mainVerticalLayout = QVBoxLayout()

        logo = QLabel()
        pixmap = QPixmap('E:/Meladee/assets/logo.png').scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)

        title = QLabel("Meladee")
        title.setFont(create_bold_font(32))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #3A0073;")

        file_name = os.path.basename(self.file_path)
        subtitle = QLabel(file_name)
        subtitle.setFont(create_bold_font(15))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #3A0073;")

        mainVerticalLayout.addWidget(logo)
        mainVerticalLayout.addWidget(title)
        mainVerticalLayout.addWidget(subtitle)

        self.backButton = IconButton({
            'back_arrow': 'E:/Meladee/assets/back_arrow.svg',
            'back_arrow_pressed': 'E:/Meladee/assets/back_arrow_pressed.svg',
        }, 'back_arrow', self)
        self.backButton.setFixedSize(60, 60)
        self.backButton.setIconSize(QSize(60, 60))
        self.backButton.clicked.connect(self.emitBackRequest)

        mainLayout = QHBoxLayout(self)
        backButtonLayout = QVBoxLayout()
        backButtonLayout.addStretch()
        backButtonLayout.addWidget(self.backButton)
        backButtonLayout.addStretch()

        mainLayout.addLayout(backButtonLayout)
        mainLayout.addLayout(mainVerticalLayout)
        
        controlsLayout = QHBoxLayout()
        controlsLayout.addStretch()

        iconSize = QSize(30, 30)

        self.playPauseButton = IconButton({
            'play': 'E:/Meladee/assets/play.svg',
            'play_pressed': 'E:/Meladee/assets/play_pressed.svg',
            'pause': 'E:/Meladee/assets/pause.svg',
            'pause_pressed': 'E:/Meladee/assets/pause_pressed.svg',
        }, 'play', self)
        self.playPauseButton.clicked.connect(self.toggle_playback)
        self.playPauseButton.setIconSize(iconSize)

        self.currentPositionLabel = QLabel("0:00 / 0:00")
        self.currentPositionLabel.setFont(create_bold_font(18))
        self.currentPositionLabel.setStyleSheet("color: #3A0073;")

        self.volumeButton = IconButton({
            'no_volume': 'E:/Meladee/assets/no_volume.svg',
            'no_volume_pressed': 'E:/Meladee/assets/no_volume_pressed.svg',
            'low_volume': 'E:/Meladee/assets/low_volume.svg',
            'low_volume_pressed': 'E:/Meladee/assets/low_volume_pressed.svg',
            'high_volume': 'E:/Meladee/assets/high_volume.svg',
            'high_volume_pressed': 'E:/Meladee/assets/high_volume_pressed.svg',
        }, 'high_volume', self)
        self.volumeButton.clicked.connect(self.muteVolume)
        self.volumeButton.setIconSize(iconSize)
        self.updateVolumeIcon(50)

        self.volumeSlider = QSlider(Qt.Horizontal)
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setValue(50)
        self.volumeSlider.valueChanged.connect(self.onVolumeChanged)
        self.volumeSlider.setMaximumWidth(150)
        self.volumeSlider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
                background: #3A0073;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::add-page:horizontal {
                background: #fff;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
                    border: 1px solid #777;
                    width: 15px; /* Adjust the width for size */
                    height: 15px; /* Make sure height is the same as width for circle */
                    margin-top: -3px; /* Adjust margin-top for vertical center alignment */
                    margin-bottom: -3px; /* Adjust margin-bottom for vertical center alignment */
                    border-radius: 7.5px; /* Half of width or height */
                }

                QSlider::handle:horizontal:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fff, stop:1 #ddd);
                    border: 1px solid #444;
                    border-radius: 7.5px; /* Maintain circular shape on hover */
                }

                QSlider::handle:horizontal:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #aaa, stop:1 #888);
                    border: 1px solid #444;
                    border-radius: 7.5px; /* Maintain circular shape when pressed */
                }
        """)

        controlsLayout.addWidget(self.playPauseButton)
        controlsLayout.addWidget(self.currentPositionLabel)
        controlsLayout.addWidget(self.volumeButton)
        controlsLayout.addWidget(self.volumeSlider)
        controlsLayout.addStretch()
        mainVerticalLayout.addLayout(controlsLayout)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.LowLatency)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.mediaPlayer.positionChanged.connect(self.on_positionChanged)

        audio = AudioSegment.from_file(file_path)
        raw_data = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            raw_data = raw_data[::2]

        self.samples = raw_data
        self.sample_rate = audio.frame_rate

        waveformFrame = QFrame()
        waveformFrame.setFixedSize(800 + 2*2, 150 + 2*2)
        waveformFrame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #3A0073;
                border-radius: 5px;
            }
        """)

        self.waveformPlot = pg.PlotWidget(waveformFrame)
        self.waveformPlot.setGeometry(2, 2, 800, 150)
        self.waveformPlot.setBackground('w')
        self.waveformPlot.setRange(yRange=[-32768, 32767])
        self.waveformPlot.setLimits(xMin=0, xMax=len(self.samples), yMin=-32768, yMax=32767)
        self.waveformPlot.getAxis('bottom').setHeight(0)
        self.waveformPlot.getAxis('left').setWidth(0)
        self.waveformPlot.setMouseEnabled(x=False, y=False)
        self.waveformPlot.hideButtons()
        self.playedWaveform = self.waveformPlot.plot(pen=pg.mkPen('#710CCF', width=1))
        self.unplayedWaveform = self.waveformPlot.plot(pen=pg.mkPen('#3A0073', width=1))
        self.unplayedWaveform.setData(self.samples)

        waveformLayout = QHBoxLayout()
        waveformLayout.addStretch()
        waveformLayout.addWidget(waveformFrame)
        waveformLayout.addStretch()
        
        mainVerticalLayout.addLayout(waveformLayout)

        classificationSimilarityLabel = QLabel("Classification Similarity:")
        classificationSimilarityLabel.setFont(create_bold_font(18))
        classificationSimilarityLabel.setStyleSheet("color: #3A0073;")
        classificationSimilarityLabel.setAlignment(Qt.AlignCenter)
        mainVerticalLayout.addWidget(classificationSimilarityLabel)

        predicted_genre_text = QLabel(f"Predicted Genre: {self.predicted_genre}")
        predicted_genre_text.setFont(create_bold_font(16))
        predicted_genre_text.setStyleSheet("color: #3A0073;")
        predicted_genre_text.setAlignment(Qt.AlignCenter)
        mainVerticalLayout.addWidget(predicted_genre_text)

    @staticmethod
    def preprocess_audio(file_path):
        y, sr = librosa.load(file_path, sr=None)
        features = ResultWindow.extract_features(y, sr, file_path)
        features = features.reshape(1, -1)
        return features
        
    def get_genre_label(index):
        genre_labels = [
            "Ambient",
            "Ambient Electronic",
            "Avant-Garde",
            "Downtempo",
            "Drone",
            "Electroacoustic",
            "Electronic",
            "Experimental",
            "Experimental Pop",
            "Field Recordings",
            "Folk",
            "Garage",
            "Glitch",
            "Hip-Hop",
            "IDM",
            "Improv",
            "Indie-Rock",
            "Industrial",
            "Instrumental",
            "Jazz",
            "Lo-Fi",
            "Musique Concrete",
            "Noise",
            "Pop",
            "Psych-Folk",
            "Punk",
            "Rock",
            "Singer-Songwriter",
            "Sound Art",
            "Soundtrack",
            "Techno",
            "Trip-Hop"
        ]
        return genre_labels[index]

    @staticmethod
    def columns():
        feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                             tonnetz=6, mfcc=20, rmse=1, zcr=1,
                             spectral_centroid=1, spectral_bandwidth=1,
                             spectral_contrast=7, spectral_rolloff=1)
        moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

        columns = []
        for name, size in feature_sizes.items():
            for moment in moments:
                it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
                columns.extend(it)

        names = ('feature', 'statistics', 'number')
        columns = pd.MultiIndex.from_tuples(columns, names=names)

        return columns.sort_values()

    @staticmethod
    def extract_features(file_path, sr):
        features = pd.Series(index=ResultWindow.columns(), dtype=np.float32)

        x, sr = librosa.load(file_path, sr=None, mono=True)

        return features

    @staticmethod
    def predict_genre(file_path, sr):
        features = ResultWindow.extract_features(file_path, sr)
        if features.shape[0] != 518:
            raise ValueError("Extracted features do not have the correct shape (518,).")

        prediction = model.predict(features.values.reshape(1, -1))
        
        max_prob_index = np.argmax(prediction)
        predicted_genre_label = ResultWindow.get_genre_label(max_prob_index)

        return predicted_genre_label

    def emitBackRequest(self):
        self.backRequested.emit()

    def toggle_playback(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playPauseButton.icon_key = 'play'
            self.playPauseButton.hover_icon_key = 'play_pressed'
            self.playPauseButton.set_icon(self.playPauseButton.icon_key)
        else:
            self.mediaPlayer.play()
            self.playPauseButton.icon_key = 'pause'
            self.playPauseButton.hover_icon_key = 'pause_pressed'
            self.playPauseButton.set_icon(self.playPauseButton.icon_key)

    def updateVolumeIcon(self, volume):
        if volume == 0:
            self.volumeButton.update_icon_keys('no_volume')
        elif volume < 50:
            self.volumeButton.update_icon_keys('low_volume')
        else:
            self.volumeButton.update_icon_keys('high_volume')

    def onVolumeChanged(self, value):
        self.mediaPlayer.setVolume(value)
        self.updateVolumeIcon(value)

    def muteVolume(self):
        if self.mediaPlayer.volume() == 0:
            self.mediaPlayer.setVolume(50)
            self.volumeSlider.setValue(50)
        else:
            self.mediaPlayer.setVolume(0)
            self.volumeSlider.setValue(0)
        self.updateVolumeIcon(self.mediaPlayer.volume())

    def updatePlaybackPosition(self, position):
        proportion = position / self.mediaPlayer.duration()
        played_samples = int(proportion * len(self.samples))

        self.playedWaveform.setData(self.samples[:played_samples])
        self.unplayedWaveform.setData(x=np.arange(played_samples, len(self.samples)), y=self.samples[played_samples:])

    def on_positionChanged(self, position):
        current_time = QTime(0, 0).addMSecs(position)
        total_time = QTime(0, 0).addMSecs(self.mediaPlayer.duration())
        self.currentPositionLabel.setText(current_time.toString('mm:ss') + " / " + total_time.toString('mm:ss'))
        self.updatePlaybackPosition(position)


    def on_durationChanged(self, duration):
        total_time = QTime(0, 0).addMSecs(duration)
        self.currentPositionLabel.setText("0:00 / " + total_time.toString('mm:ss'))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())