class Config_zap():
    def __init__(self):
        self.dataset = 'zap50k_shoes-reid'
        self.log_path = "log/" #log dir and saved model dir
        self.model_path = 'save_model/'
        self.att_path = './processed_data/zap50k/'
        self.resume = ''
        
        self.gpu = "7,8"
        self.num_workers = 8

        self.save_epoch = 40  
         
        self.low_dim = 2048
        self.img_w = 96
        self.img_h = 96      
        self.num_att = 30
        
        self.batch_size = 32
        self.test_batch = 32
        self.num_instance_per_id = 1
        
        self.optimizer = 'Adam'
        self.lr = 0.00035



