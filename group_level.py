                
if __name__ == '__main__':

    # wandb.init(project='rvos',
    #            group='a2ds_final',
    #            name='A2DS dataset visualization')
    # root='/home/xhh/datasets/a2d_sentences'
    # from datasets.rvos.a2ds import visualize_dataset_information
    # visualize_dataset_information(root)
    
    # YRVOS_perWindow_perExp    
    from data_schedule.rvos.yrvos import show_dataset_information_and_validate
    root = '/home/xuhuihui/datasets/youtube_rvos'
    show_dataset_information_and_validate(root)