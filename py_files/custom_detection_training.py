from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=r"Z:\DeepLabCut\misc\tensorflow\ImageAI\project\data")
trainer.setTrainConfig(object_names_array=["White_mouse", "Black_mouse"], batch_size=10, num_experiments=100, train_from_pretrained_model="detection_model-ex-025--loss-4.182.h5")
trainer.trainModel()