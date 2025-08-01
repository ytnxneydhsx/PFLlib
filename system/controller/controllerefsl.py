import controller.controllerbase as controllerbase


class controllerefsl(controllerbase):
    def __init__(self, args,get_server_resource_method, get_client_resource_method):
        super().__init__(args)
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.global_model = args.global_model
        self.server_resource= get_server_resource_method()
        self.client_resource_list= get_client_resource_method()
        
        
        def get_server_resource(self):
            
