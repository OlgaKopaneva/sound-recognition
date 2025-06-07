# Распознавание звуков  
### Автор: Копанева Ольга  

## Постановка задачи  

Задача - создание системы теггирования аудио, способной распознать звук в поданном на вход модели аудиофрагменте и сопоставить ему категорию из списка 41 различных вариантов, взятых из онтологии [Google AudioSet](https://research.google.com/audioset////////ontology/index.html) . Предполагается распознавание таких звуков, как: музыкальные инструменты, звуки людей, бытовые звуки и звуки животных.  

Конечная цель системы – автоматизация процессов аннотирования звуковых коллекций и создания титров для неречевых событий в аудиовизуальном контенте.  

Задача была представлена как соревнование Kaggle: [«Freesound General-Purpose Audio Tagging Challenge»](https://www.kaggle.com/competitions/freesound-audio-tagging/overview)  

## Формат входных и выходных данных  

- **на вход** модели подается аудиофайл формата WAV (PCM) 16 бит, 44,1 кГц, mono (длительностью в среднем от 300 мс до 30 с);  
- **на выходе** получаем тег из списка:  
"Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing".  

## Метрики  

В соревновании для оценки результатов использовалась метрика **Mean Average Precision @ 3 (mAP@3)** (принимает до трех ранжированных предсказанных меток для каждого аудиоклипа и дает полную оценку, если правильная метка оказывается первой, с меньшей оценкой за правильные предсказания меток на втором или третьем месте). Т.к. данная метка часто используется для измерения точности детекции, то она подходит к поставленной задаче.  

Baseline, описанный в статье , достигает mAP@3 = 70%
