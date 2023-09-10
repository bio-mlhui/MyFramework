                
if __name__ == '__main__':

    # wandb.init(project='rvos',
    #            group='a2ds_final',
    #            name='A2DS dataset visualization')
    root='/home/xhh/datasets/a2d_sentences'
    from datasets.rvos.a2ds import visualize_dataset_information
    visualize_dataset_information(root)
    
    # # YRVOS_perWindow_perExp    
    # from datasets.rvos.yrvos import show_dataset_information_and_validate
    # import wandb
    # root = '/home/xhh/datasets/youtube_rvos'
    # wandb.init(project='rvos',
    #             group='yrvos_final',
    #             name='Youtube-RVOS dataset visualization',
    #             id='rvos_yrvos_final_datasetVis',
    #             resume='must')
    # show_dataset_information_and_validate(root)