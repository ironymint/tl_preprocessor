# tl_preprocessor
Just add a few sample images per class for transfer learning and run [tl_main.py]. It classifies tons of images as a magic

# Install
Tensorflow==2.4.1<br />
pip install -r requirements.txt

# How to use
1. Put a few samples(at least 16) on [./dataset/<dataset_name>/train/<class_names>/]
2. Put images that you want to classify on [./dataset/<dataset_name>/test/]
3. run "python tl_main.py --path ./dataset/<dataset_name>/"
4. check results on [./dataset/<dataset_name>/test_classified/]