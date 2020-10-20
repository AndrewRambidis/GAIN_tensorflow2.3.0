This is the code for the GAIN algorithm by Jinsung Yoon et.al. recoded to work with tensorflow 2

This does not use the keras API

export your python path to accept the tools directory:
    export PYTHONPATH=/path/to/src:$PYHTONPATH

**usage:**
 
For quick usage enter: python main_letter_spam.py --data_name letter

**More Details:**

main_letter_spam.py [-h] [--data_name {letter,spam}]
                           [--miss_rate MISS_RATE] [--batch_size BATCH_SIZE]
                           [--hint_rate HINT_RATE] [--alpha ALPHA]
                           [--iterations ITERATIONS]

optional arguments:
  -h, --help            show this help message and exit

  --data_name {letter,spam}

  --miss_rate MISS_RATE     missing data probability

  --batch_size BATCH_SIZE   number of samples in mini-batch

  --hint_rate HINT_RATE     hint probability

  --alpha ALPHA     hyperparameter

  --iterations ITERATIONS    number of training iterations
