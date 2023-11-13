# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['GUI_infection_final.py'],
             pathex=[],
             binaries=[],
             datas=[(r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\3_channel_best.hdf5', '.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\Augmented_TverskyLoss.h5', '.'),
		(r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\9k-Aug_150ep.hdf5', '.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\Augmented_TverskyLoss_best.hdf5','.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\Augmented_10_times_BinaryCrossentropy_norm_every_slice_fixed_drop0_best.hdf5','.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\Augmented_10_times_TverskyLoss_norm_every_slice_fixed_best.hdf5','.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\Augmented_binary_focal_loss_norm_every_slice_fixed_drop1_best.hdf5','.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\Augmented_BinaryCrossentropy_norm_every_slice_fixed_best.hdf5','.'),
                (r'C:\Users\Alireza\OneDrive - University of Oklahoma\Documents\GUI\AttRes_2Data_TverskyLoss_Augmented_every_slice_fixed_D1_best.hdf5','.')],
             hiddenimports=["pydicom.encoders.gdcm", "pydicom.encoders.pylibjpeg", "fastremap"],
             hookspath=["MRCNN\mrcnn", "MRCNN\mrcnn\config.py"],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='Infection_segmentation',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Infection_segmentation_new')
