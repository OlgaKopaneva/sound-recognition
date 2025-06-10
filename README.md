# Распознавание звуков

#### Автор: Копанева Ольга

## Постановка задачи

Задача - создание системы теггирования аудио, способной распознать звук в
поданном на вход модели аудиофрагменте и сопоставить ему категорию из списка 41
различных вариантов, взятых из онтологии
[Google AudioSet](https://research.google.com/audioset////////ontology/index.html)
. Предполагается распознавание таких звуков, как: музыкальные инструменты, звуки
людей, бытовые звуки и звуки животных.

Конечная цель системы – автоматизация процессов аннотирования звуковых коллекций
и создания титров для неречевых событий в аудиовизуальном контенте.

Задача была представлена как соревнование Kaggle:
[«Freesound General-Purpose Audio Tagging Challenge»](https://www.kaggle.com/competitions/freesound-audio-tagging/overview)

## Формат входных и выходных данных

- **на вход** модели подается аудиофайл формата WAV (PCM) 16 бит, 44,1 кГц, mono
  (длительностью в среднем от 300 мс до 30 с);
- **на выходе** получаем тег из списка: "Acoustic_guitar", "Applause", "Bark",
  "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
  "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
  "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping",
  "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
  "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
  "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum",
  "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
  "Writing".

## Метрики

В соревновании для оценки результатов использовалась метрика **Mean Average
Precision @ 3 (mAP@3)** (принимает до трех ранжированных предсказанных меток для
каждого аудиоклипа и дает полную оценку, если правильная метка оказывается
первой, с меньшей оценкой за правильные предсказания меток на втором или третьем
месте). Т.к. данная метка часто используется для измерения точности детекции, то
она подходит к поставленной задаче.

Baseline, описанный в статье[1], достигает **mAP@3 = 70%**

Максимальный результат на Leaderboard достигает **mAP@3 = 95,3%**

[Решение](https://www.kaggle.com/code/daisukelab/freesound-dataset-kaggle-2018-solution/notebook),
которое взято за основу проекта, выдает **mAP@3 = 91,7%**

[1]: Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier
Favory, Jordi Pons, Xavier Serra. "General-purpose Tagging of Freesound Audio
with AudioSet Labels: Task Description, Dataset, and Baseline". Proceedings of
the DCASE 2018 Workshop (2018) -
[https://arxiv.org/abs/1807.09902](https://arxiv.org/abs/1807.09902)

## Данные

Будем использовать датасет указанного выше соревнования
[«Freesound General-Purpose Audio Tagging Challenge»](https://www.kaggle.com/competitions/freesound-audio-tagging/data).

Актуальная ссылка на датасет:
[http://zenodo.org/records/2552860#.XFD05fwo-V4](http://zenodo.org/records/2552860#.XFD05fwo-V4)

Суммарно датасет содержит 11073 аудиофайлов (9473 train, 1600 test) с
присвоенными тегами (при этом для 3710 фрагментов метки проверены вручную, т.е.
достоверны, остальные размечены автоматически и могут содержать ошибки, уровень
точности – 60-70% согласно описанию датасета). Все аудиосэмплы в этом наборе
данных взяты из [Freesound](https://freesound.org/) и представлены в виде
несжатых аудиофайлов формата WAV (PCM) 16 бит, 44,1 кГц, mono, длительность
каждого фрагмента от 300 мс до 30 с.

**Потенциальная проблема** – неточность изначально присвоенных меток, что
необходимо учитывать в работе.

## Моделирование

### Бейзлайн

Baseline, описанный в статье [1], предполагает использование 3-х слойной CNN,
которой на вход подается преобразованный в log-mel спектрограмму звук, и в конце
слой softmax для классификации к одному из 41 тегов. Чтобы не ошибиться в
переводе терминологии, описание преобразований исходного звука приведу в формате
цитаты из указанной статьи: _«Incoming audio is divided into overlapping windows
of size 0.25s with a hop of 0.125s. These windows are decomposed with a
short-time Fourier transform using 25ms windows every 10ms. The resulting
spectrogram is mapped into 64 mel-spaced frequency bins, and the magnitude of
each bin is log-transformed after adding a small offset to avoid numerical
issues. Predictions are obtained for a clip of arbitrary length by running the
model over 0.25s-wide windows every 0.125s, and averaging all the window-level
predictions to obtain a clip-level prediction»_

### Основная модель

Основная модель будет базироваться на ноутбуке:
[https://www.kaggle.com/code/daisukelab/freesound-dataset-kaggle-2018-solution](https://www.kaggle.com/code/daisukelab/freesound-dataset-kaggle-2018-solution)

Предложенная в ноутбуке стратегия решения представляет собой **ансамбль двух
подходов:**

1. LH: используется самое высокое качество, но только для начальной части
   звуков;
2. X: выделяются из образцов достаточно длинные, но используются с менее высоким
   качеством.

Из специфического в обработке данных следует отметить использование
специализированной библиотеки
[librosa](https://librosa.org/doc/latest/index.html), служащей для работы с
аудио и музыкой. Именно эта библиотека строит спектрограммы.

Сама модель обучения в исходном ноутбуке использует keras, **переложена в рамках
проекта на pytorch.** Архитектура модели базируется на AlexNet (соревнование и,
соответственно, решение достаточно старые).

## Setup

Для управления зависимостями использован
[poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

Необходимые зависимости прописаны в файле [pyproject.toml](pyproject.toml).

Для установки зависимостей достаточно установить poetry по инструкции:
[https://python-poetry.org/docs/#installing-with-the-official-installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
и выполнить `poetry install`

## Train

**Получение данных:** Данные хранятся в локальном хранилище, в репозитории их
dvc образы.

Для подтягивания данных (если доступно локальное хранилище): `dvc pull` Если не
доступно: `python download_data.py` (скачает из исходного расположения и
распакует)

**Обучение:**

_Первый подход (X-подход)_ (описание - см. выше в разделе про модель)

Перед запуском зайти в [configs\model\model.yaml](configs\model\model.yaml) и
включить соответсвующую размерность сети

`python sound_recognition/train.py data_loading@preprocessing=preprocessing_x training=training_x`

_Второй подход (LH-подход)_

Перед запуском зайти в [configs\model\model.yaml](configs\model\model.yaml) и
включить соответсвующую размерность сети

`python sound_recognition/train.py data_loading@preprocessing=preprocessing_lh training=training_lh`

Исходный проект предполагает в последствии смешать результаты обоих подходов
(для этого можно воспользоваться
[sound_recognition\ensamble.py](sound_recognition\ensamble.py)), но так как с
технической стороной обучения этот процесс уже не связан, то такой запуск не
проводился.

## Production preparation

Веса моделей сохранены в папке [models](models) для дальнейшего использования
