import gdown
import os

path = os.getcwd()[:-7]

gdown.download(
    'https://drive.google.com/uc?id=1QoW3pGIJYx4wP1BWUi2sBHDZkWb_0jVp',
    f'{path}/models/best_model_state.bin'
)