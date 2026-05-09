class Config:
    DATA_DIR = "data/"
    TRAIN_A = "data/real"
    TRAIN_B = "data/cartoon"
    
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    EPOCHS = 2
    
    LR = 0.0002
    LAMBDA_CYCLE = 10
    LAMBDA_IDENTITY = 5

    DEVICE = "cpu"