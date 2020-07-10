class Config_market():
    def __init__(self):
        
        self.dataset = 'Market-1501'
        self.log_path = "log/" #log dir and saved model dir
        self.raw_att = '/mnt/nasbi/no-backups/datasets/reid_dataset/market1501/Attribute/market_attribute.mat'
        self.model_path = 'save_model/'
        self.data_dir = '/mnt/nasbi/no-backups/datasets/reid_dataset'
        self.att_path = './processed_data/market/market_attributes.pkl'
        self.resume = ''
        
        self.gpu = "7,8"
        self.num_workers = 8

        self.save_epoch = 40  
         
        self.low_dim = 2048
        self.img_w = 128
        self.img_h = 384   
        self.num_att = 8
        
        self.batch_size = 16 # change to 128 during training market
        self.test_batch = 16
        self.num_instance= 1
        
        self.optimizer = 'Adam'
        self.lr = 3e-4