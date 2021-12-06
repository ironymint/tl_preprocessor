# tl_preprocessor
Just add a few sample images per class and run "python tl_main.py"(For Transfer Learning).<br/>
It classifies tons of images like magic.

# Install
Tensorflow==2.4.1<br/>
pip install -r requirements.txt

# How to use
1. Put a few samples(at least 16) on ["/path/to/dataset/train/<class_name_1>/" , "/path/to/dataset/train/<class_name_2>/", ...]
2. Put images that you want to classify on "/path/to/dataset/test/imgs/"
3. Run "python tl_main.py --path /path/to/dataset/"
4. Check results on "/path/to/dataset/test_classified/"