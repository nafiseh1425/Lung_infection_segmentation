# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['GUI_severity_segmentation.py'],
             pathex=[],
             binaries=[],
             datas=[(r'C:\Users\OneDrive - University of Oklahoma\Documents\GUI\3_channel_best.hdf5', '.'),
		(r'C:\Users\OneDrive - University of Oklahoma\Documents\GUI\9k-Aug_150ep.hdf5', '.')],
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
          name='GUI_5-2-All_new',
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
               name='GUI_5-2-All_new')
