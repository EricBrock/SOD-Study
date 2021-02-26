# Common Code for Salient Object Detection



## Usage



1. ### model evaluation

   ```python
   import fjn_util
   flops, params, size = sod_util.get_flops_params_size(model=your_model, input_size=(3,224,224), as_strings=True, print_per_layer_stat=False) 
   
   print(flops)
   print(params)
   print(size)
   ```

2. ### sod evaluation
   ```python
   import fjn_util
   img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
   img2 = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)
   
   sod_index = git_util.SOD_Index(y_pred=img1, y_true=img2,binarization=True)
   
   print(sod_index.cal_precision())
   print(sod_index.cal_recall())
   print(sod_index.cal_miou())
   print(sod_index.cal_mae())
   print(sod_index.cal_fmeasure())
   print(sod_index.cal_smeasure())
   sod_index.cal_prc(True)
   ```
