QMODE_FIXEDHU = 0
QMODE_STAT = 1

def enforceGLCMQuantizationMode(feature_def, modality):
    # print(feature_def.args, modality)
    new_def = feature_def.copy()
    # force stat based GLCM quantization if not CT image
    if 'gray_levels' in new_def.args and 'binwidth' in new_def.args:
        if modality and modality.lower() == 'ct':
            # enforce glcm FIXEDHU based quantization
            del new_def.args['gray_levels']
        else:
            # force glcm STAT based quantization
            del new_def.args['binwidth']
    # elif 'binwidth' in new_def.args and modality and modality.lower() != 'ct':
    #     # force glcm STAT based quantization
    #     del new_def.args['binwidth']
    return new_def
