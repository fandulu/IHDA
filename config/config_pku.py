class Config_pku():
    def __init__(self):
        self.dataset = 'PKU-Sketch-reid'
        self.data_dir = '/mnt/nasbi/no-backups/datasets/reid_dataset/PKUSketchRE-ID/'
        self.log_path = "log/" #log dir and saved model dir
        self.model_path = 'save_model/'
        self.att_path = './processed_data/market/market_attributes.pkl'
        self.resume = 'market.t'
        
        self.test_mode = 'sketch -> color'
        
        self.gpu = "5,6"
        self.num_workers = 8

        self.save_epoch = 40  
         
        self.low_dim = 2048
        self.img_w = 128
        self.img_h = 384   
        self.num_att = 8
        
        self.batch_size = 48
        self.test_batch = 48
        self.num_instance = 2
        
        self.optimizer = 'Adam'
        self.lr = 3e-4



