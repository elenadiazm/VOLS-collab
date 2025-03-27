# Script for imaging the VOLS observations at C-band
# First self-calibration is performed spw by spw on the pointings that contain bright sources
# Second self-calibration is performed (combining all spws) on the pointings that contain sources at 10sigma

# Last update 2025.03 .- Elena Díaz-Márquez

import os
import sys
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u  
import bdsf
import glob

# NOTE.- Change this in case you are running the script in a different server

#my_dir = '/share/Part2/ediaz/VOLS/'     # multivac
my_dir = '/home/VOLS/'                    # servervols


# User-defined parameters

n_ant = 25 # number of antennas
t_int = 32  # integration time in seconds
t_exp = 2040  # on-source observing time in seconds (34 minutes)

my_calmode = 'p'

# Set the observation frequency in GHz (C-band)

freq = 3.955972 # in GHz --  Minimum frequency, corresponding to the maximum FoV
theta_pb = 42 / freq  # arcminutes
beam_radius = (theta_pb / 60) / 2  # beam radius in degrees

# Measurement sets to image

# my_visFileBaseName = ['22A-195.sb41668223.eb41752682.59672.87566575232_cont',
# '22A-195.sb41668223.eb41756347.59679.919291342594_cont',
# '22A-195.sb41668223.eb41763155.59682.964581215274_cont',
# '22A-195.sb41668223.eb41771911.59684.805362025465_cont',
# '22A-195.sb41668223.eb41774359.59688.783021979165_cont',
# '22A-195.sb41668223.eb41776239.59690.95766400463_cont',
# '22A-195.sb41668223.eb41784557.59692.92574443287_cont',
# '22A-195.sb41668223.eb41784722.59693.823116562504_cont',
# '22A-195.sb41668223.eb41784724.59693.95610815972_cont',
# '22A-195.sb41668223.eb41785635.59695.81217344907_cont',
# '22A-195.sb41668223.eb41788325.59698.93521715278_cont',
# '22A-195.sb41668223.eb41788343.59699.900189837965_cont',
# '22A-195.sb41668223.eb41788359.59700.754512060186_cont',
# '22A-195.sb41668223.eb41788361.59700.88744737269_cont',
# '22A-195.sb41668223.eb41788874.59701.92147296296_cont',
# '22A-195.sb41668223.eb41789898.59702.90341413194_cont',
# '22A-195.sb41668223.eb41815091.59721.7578799537_cont',
# '22A-195.sb41668223.eb41818503.59723.85374815972_cont',
# '22A-195.sb41668223.eb41837135.59733.80435298611_cont',
# '22A-195.sb41668223.eb41838351.59734.75052173611_cont',
# '22A-195.sb41668223.eb41842850.59737.742964062505_cont',
# '22A-195.sb41668223.eb41848883.59740.76904108796_cont',
# '22A-195.sb41668223.eb41849717.59741.79960082176_cont',
# '22A-195.sb41668223.eb41852333.59744.63347153935_cont',
# '22A-195.sb41668223.eb41852443.59744.76640462963_cont',
# '22A-195.sb41668223.eb41905952.59761.608695324074_cont']

my_visFileBaseName = ['22A-195.sb41668223.eb41752682.59672.87566575232_cont']

my_vislist = [basename + '.ms' for basename in my_visFileBaseName]

my_dates = ['20220403','20220410','20220413','20220415','20220419','20220421','20220423','20220424a','20220424b','20220426','20220429','20220430',
       '20220501a','20220501b','20220502','20220503','20220522','20220524','20220603','20220604','20220607', '20220610','20220611','20220614a','20220614b','20220701']

# Submosaics to image

my_submosaics = ['00']

# CLEANing is split into different submosaics

# my_submosaicData: dictionary containing parameters for submosaics:
# - my_submosaicPointings : pointings included in each submosaic
# - my_submosaicPhaseCenter : phase centers of each submosaic
# - my_submosaicImsize : image sizes of each submosaic

my_submosaicData = {
'my_submosaicPointings': {
'00': 'P108,P90,P69,P48,P27,P6,P5,P26,P47,P68,P89,P88,P67,P46,P25,P4,P3,P24,P107,P87,P66,P45,P65,P44,P23,P2,P1,P22,P43,P64,P85,P86,P106,P91,P70,P49,P109,P92,P71,P50,P29,P8,P7,P28',
'01': 'P109,P92,P71,P50,P29,P8,P30,P9,P31,P10,P11,P32,P53,P74,P95,P94,P110,P93,P72,P51,P73,P52,P54,P33,P12,P13,P34,P55,P76,P97,P98,P77,P56,P35,P11,P96,P75,P112,P14,P111',
'02':'P98,P77,P56,P35,P112,P99,P78,P57,P36,P15,P37,P16,P17,P38,P59,P80,P101,P102,P81,P113,P100,P79,P58,P60,P39,P18,P19,P40,P61,P82,P103,P104,P83,P62,P41,P20,P21,P114,P115,P105,P84,P63,P42,P14',
},

'my_submosaicPhaseCenter': {
'00': 'ICRS 05:35:10.465 -05.44.45.0',
'01': 'ICRS 05:35:10.465 -05.22.45.0',
'02': 'ICRS 05:35:10.465 -04.59.45.0'
},

'my_submosaicImsize': {                       
'00': [16500, 18000],                       
'01': [16500, 18000],                     
'02': [16500, 18000],
},

'my_uvrange':{
'00':'>35klambda',
'01': '>100klambda', # NOTE.- We will only use this when it is needed to create a mask, in order to avoid extended emission
'02': '>35klambda',
},

'my_rms':{  # NOTE.- NEED TO COPY THEM 
'00': my_dir + 'regions/measure-rms-00.crtf',
'01':my_dir + 'regions/measure-rms-01.crtf',
'02':my_dir + 'regions/measure-rms-02.crtf',
}
}

for i in range(0, len(my_vislist)):

    my_visFile = my_dir + 'CALIBRATED_CONTINUUM_SPW_ORDERED/' + my_vislist[i]

    print('::: VOLS ::: ... Processing the measurement set ' + my_vislist[i])

    # delmod(vis=my_visFile, otf= False)

    all_pointings = {f'P{j}' for j in range(1, 116)} 

    for my_submosaic in my_submosaics:

        #log_file = 'imagin-selfcal.' + str(my_dates[i]) + '.' +str(my_submosaic)+ '.log'

        #original_stdout = sys.stdout
        #sys.stdout = open(log_file,'w')

        # selfcal_pointings = set()  # Setting unique values for the self-calibrated pointings -- ON EACH SUBMOSAIC ! -- I know this cant be efficient if we have a source on a pointing that is in submosaic '00' and '01' for example,
        #                             # but i am not sure how to take into account the pointings that are self-calibrated in other submosaics

        # print('::: VOLS ::: ... Submosaic ' + str(my_submosaic) + ' centered at ' + str(my_submosaicData['my_submosaicPhaseCenter'][my_submosaic]))

        my_spws = '2~31' # NOTE.- Use this in case your spw's are sorted by frequency (see script-order-spw.py)

        print('==> Using spws ' + my_spws + ' for the imaging')

        # os.system('mkdir -p ' + my_dir + 'images/dirty')

        # my_imageFile = my_dir + 'images/dirty/VOLS_dirty_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)

        # os.system('rm -rf ' + my_imageFile + '.*')

        # print('::: VOLS ::: ... Creating a dirty image')

        # tclean(vis=my_visFile,
        #            uvrange=my_submosaicData['my_uvrange'][my_submosaic], 
        #            datacolumn='data',
        #            spw=my_spws,
        #            field=my_submosaicData['my_submosaicPointings'][my_submosaic],
        #            phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
        #            imagename=my_imageFile,
        #            imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
        #            cell=['0.125arcsec'],
        #            stokes='I',
        #            specmode='mfs',
        #            gridder='mosaic',
        #            mosweight=False,
        #            usepointing=False,
        #            pblimit=0.1,
        #            deconvolver='mtmfs',
        #            nterms=2,
        #            restoration=True,
        #            pbcor=False,
        #            weighting='briggs',
        #            robust=0.5,
        #            npixels=0,
        #            niter=0,
        #            #usemask='user',
        #            #mask=my_maskFile,
        #            #threshold='0.1mJy',
        #            interactive=False,
        #            restart=False,
        #            savemodel='none',
        #            calcres=True,
        #            calcpsf=True,
        #            parallel=False,
        #            pbmask=0.0,
        #            )
        # os.system('cp -r ' + my_imageFile + '.image.tt0 .')

        # print('::: VOLS ::: ... Measuring the peak of the image')

        # dirty_stats = imstat(imagename= my_imageFile +'.image.tt0')
        # peak = dirty_stats['max'][0]

        # print('==> Peak: '+ str(peak) + ' Jy/beam')

        # print('::: VOLS ::: ... Setting a threshold of 80 percent of the peak')

        # threshold = 0.8 * peak

        # print('::: VOLS ::: ... Creating the mask for submosaic ' + str(my_submosaic) + ' using the dirty image')

        # os.system('mkdir -p ' + my_dir + 'masks/dirty')

        # my_maskFile = my_dir + 'masks/dirty/VOLS_dirty_mask_0.8peak_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)

        # os.system('rm -rf ' + my_maskFile + '.*')

        # # NOTE.- not sure why but immath is not working if the image is not in the same directory.

        # immath(
        #             imagename='VOLS_dirty_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) +'.image.tt0',
        #             expr='iif(IM0 >' + str(threshold) + ', 1.0, 0.0)',
        #             outfile=my_maskFile + '.mask'
        #         )

        # os.system('rm -r VOLS_dirty_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)  + '.image.tt0')

        # print('::: VOLS ::: ... Creating a clean image')

        # os.system('mkdir -p ' + my_dir + 'images/clean')

        # my_imageFile = my_dir + 'images/clean/VOLS_shallow_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)
        # my_maskFile = my_dir + 'masks/dirty/VOLS_dirty_mask_0.8peak_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) # already defined

        # os.system('rm -rf ' + my_imageFile + '.*')

        # tclean(vis=my_visFile,
        #           uvrange=my_submosaicData['my_uvrange'][my_submosaic], 
        #           datacolumn='data', 
        #           spw=my_spws,
        #           field=my_submosaicData['my_submosaicPointings'][my_submosaic],
        #           phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
        #           imagename=my_imageFile,
        #           imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
        #           cell=['0.125arcsec'],
        #           stokes='I',
        #           specmode='mfs',
        #           gridder='mosaic',
        #           mosweight=False,
        #           usepointing=False,
        #           pblimit=0.1,
        #           deconvolver='mtmfs',
        #           nterms=2,
        #           restoration=True,
        #           pbcor=False,
        #           weighting='briggs',
        #           robust=0.5,
        #           npixels=0,
        #           niter=10000000,
        #           usemask='user',
        #           mask=my_maskFile + '.mask',
        #           threshold='0.1mJy',
        #           interactive=False,
        #           restart=False,
        #           savemodel='none',
        #           calcres=True,
        #           calcpsf=True,
        #           parallel=False,
        #           pbmask=0.0,
        #           )

        # os.system('cp -r ' + my_imageFile + '.image.tt0 .')

        # print('::: VOLS ::: ... Measuring the Median Absolute Deviation (MAD)')

        # shallow_clean_stats = imstat(imagename= my_imageFile +'.image.tt0')
        # mad = shallow_clean_stats['medabsdevmed'][0]

        # print('==> MAD: '+ str(mad) + ' Jy/beam')

        # print('::: VOLS ::: ... Setting a threshold of MAD x 1.5 x 20, similar to a 20sigma threshold')

        # threshold = mad*1.5*20

        # print('::: VOLS ::: ... Creating the mask for submosaic ' + str(my_submosaic) + ' using the clean image')

        # os.system('mkdir -p ' + my_dir + 'masks/clean')

        # my_maskFile = my_dir + 'masks/clean/VOLS_clean_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)

        # os.system('rm -rf ' + my_maskFile + '.*')

        # # NOTE.- not sure why but immath is not working if the image is not in the same directory.

        # immath(
        #             imagename='VOLS_shallow_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) +'.image.tt0',
        #             expr='iif(IM0 >' + str(threshold) + ', 1.0, 0.0)',
        #             outfile=my_maskFile + '.mask'
        #         )

        # os.system('rm -r VOLS_shallow_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)  + '.image.tt0')

        # print('::: VOLS ::: ... Creating a clean image using the mask')

        # os.system('mkdir -p ' + my_dir + 'images/clean')

        # my_imageFile = my_dir + 'images/clean/VOLS_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)
        # my_maskFile = my_dir + 'masks/clean/VOLS_clean_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) # already defined

        # os.system('rm -rf ' + my_imageFile + '.*')

        # tclean(vis=my_visFile,
        #           uvrange='>35klambda', # we do NOT need to create a mask 
        #           datacolumn='data', 
        #           spw=my_spws,
        #           field=my_submosaicData['my_submosaicPointings'][my_submosaic],
        #           phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
        #           imagename=my_imageFile,
        #           imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
        #           cell=['0.125arcsec'],
        #           stokes='I',
        #           specmode='mfs',
        #           gridder='mosaic',
        #           mosweight=False,
        #           usepointing=False,
        #           pblimit=0.1,
        #           deconvolver='mtmfs',
        #           nterms=2,
        #           restoration=True,
        #           pbcor=False,
        #           weighting='briggs',
        #           robust=0.5,
        #           npixels=0,
        #           niter=10000000,
        #           usemask='user',
        #           mask=my_maskFile + '.mask',
        #           threshold='0.1mJy',
        #           interactive=False,
        #           restart=False,
        #           savemodel='none',
        #           calcres=True,
        #           calcpsf=True,
        #           parallel=False,
        #           pbmask=0.0,
        #           )
       
        # exportfits(imagename = my_imageFile + '.image.tt0', fitsimage = my_imageFile + '.fits', overwrite = True)

        # print('::: VOLS ::: ... Checking the RMS from an emission-free region of the image')

        # clean_stats = imstat(imagename= my_imageFile +'.image.tt0', region=my_submosaicData['my_rms'][my_submosaic])

        # rms = clean_stats['rms'][0]

        # my_imageName = 'VOLS_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)

        # rms_data = pd.DataFrame([[my_imageName, rms]], columns=['observation', 'rms'])

        # if os.path.exists(my_dir + 'data/rms.csv'):
        
        #  # Load data and append

        #     rms_df = pd.read_csv(my_dir + 'data/rms.csv')
        #     rms_df = pd.concat([rms_df, rms_data], ignore_index=True)

        # else:
        #     rms_df = rms_data  # create a new file if it does not exist

        
        # rms_df.to_csv(my_dir + 'data/rms.csv', index=False)

        # print('The RMS value is stored in ' + my_dir + 'data/rms.csv')

        # print('::: VOLS ::: ... Locating the bright sources of the submosaic ' + str(my_submosaic))

        # print('::: VOLS ::: ... Running pyBDSF')

        # my_imageFits = my_imageFile + '.fits' 

        # print('==> Processing '+ my_imageFits)

        # # Run PyBDSF with the specified parameters

        # img = bdsf.process_image(my_imageFits,
        #                      adaptive_rms_box=True,
        #                      thresh_isl=10.0,
        #                      thresh_pix=10.0)
    

        # my_catalog = my_dir + 'data/' + my_imageName  + '_10sigma_catalog.csv'
        
        # os.system('rm -r ' + my_catalog)

        # # Write the source catalog

        # img.write_catalog(outfile=my_catalog,
        #             catalog_type = 'srl',
        #             format = 'csv') 

        # print('::: VOLS ::: ... Finishing pyBDSF')

        # print('==> You can check the catalog at 10sigma in ' + my_dir + 'data/' + my_imageName + '_10sigma_catalog.csv')

        # column_names = [ "Source_id", "Isl_id", "RA", "E_RA", "DEC", "E_DEC", "Total_flux", "E_Total_flux", "Peak_flux", 
        #                 "E_Peak_flux", "RA_max", "E_RA_max", "DEC_max", "E_DEC_max", "Maj", "E_Maj","Min", "E_Min", "PA", 
        #                 "E_PA", "Maj_img_plane", "E_Maj_img_plane", "Min_img_plane","E_Min_img_plane", "PA_img_plane", 
        #                 "E_PA_img_plane", "DC_Maj", "E_DC_Maj", "DC_Min","E_DC_Min", "DC_PA", "E_DC_PA", "DC_Maj_img_plane", 
        #                 "E_DC_Maj_img_plane", "DC_Min_img_plane","E_DC_Min_img_plane", "DC_PA_img_plane", "E_DC_PA_img_plane", 
        #                 "Isl_Total_flux", "E_Isl_Total_flux", "Isl_rms", "Isl_mean", "Resid_Isl_rms", "Resid_Isl_mean", "S_Code"
        #                 ]
        
        # my_catalog_df = pd.read_csv(my_catalog, comment='#', names=column_names) # NOTE.- Need to check because the first lines are with #

        # print('::: VOLS ::: ... Calculating the (S/N)_selfcal')


        # # Compute (S/N)_self condition

        # my_catalog_df['snr_self'] = (my_catalog_df['Peak_flux'] / rms) * (1/np.sqrt(n_ant - 3)) * (1/np.sqrt(t_exp/t_int))

        # coords_deg = SkyCoord(ra=my_catalog_df['RA'].values * u.degree, dec=my_catalog_df['DEC'].values * u.degree, frame='icrs')

        # my_catalog_df['RA_hour_angle'] = coords_deg.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad = True)
        # my_catalog_df['DEC_degree'] = coords_deg.dec.to_string(unit=u.degree, sep=':', precision=2, pad=True)
    

        # error_coords = SkyCoord(ra=my_catalog_df['E_RA'].values * u.degree, dec=my_catalog_df['E_DEC'].values * u.degree, frame='icrs')
        # my_catalog_df['E_RA_hour_angle'] = error_coords.ra.to_string(unit=u.hourangle, sep=':', precision=4, pad=True)
        # my_catalog_df['E_DEC_degree'] = error_coords.dec.to_string(unit=u.degree, sep=':', precision=4, pad=True)

        # print('::: VOLS ::: ... Checking the pointings that contain each source')

        # listobs = pd.read_csv('vols-listobs-cband.csv', sep=';', header=0) # NOTE.- NEED TO ADD THIS TO EACH SERVER

        # coords = SkyCoord(ra=listobs['RA'], dec=listobs['Decl'], unit=(u.hourangle, u.deg))

        # my_catalog_df['pointings'] = [[] for _ in range(len(my_catalog_df))]

        # for index, row in my_catalog_df.iterrows():
    
        #     my_source = SkyCoord(row['RA_hour_angle'], row['DEC_degree'], unit=(u.hourangle, u.deg))

        #     points_in_region = []
            
        #     for k, coord in enumerate(coords):
        #         # Calculate the separation between the current pointing and the target
        #         separation = coord.separation(my_source)

        #         # If the separation is within the beam radius, add to points_in_region
        #         if separation <= beam_radius * u.deg:
        #             points_in_region.append(listobs.iloc[k]['Name']) # storing as a list

        #     my_catalog_df.at[index, 'pointings'] = points_in_region 

        # my_catalog_df.sort_values(by=['Peak_flux'], ascending=False, inplace=True) # Sorting from higher value to lower value of the peak flux

        # my_catalog_df.to_csv(my_catalog, index=False)

        # print('::: VOLS ::: ... Creating a new dataset with the sources that accomplish (S/N) > 3') 

        # bright_sources_df = my_catalog_df[my_catalog_df['snr_self'] > 3] # NOTE.- Need to check if the first lines are with #

        # bright_sources_df.to_csv(my_dir + 'data/' + my_imageName  + '_bright_sources.csv', index=False) 

        # print('==> You can check the bright sources catalog in ' + my_dir + 'data/' + my_imageName + '_bright_sources.csv')

        # print('::: VOLS ::: ... Working first with the bright sources')

        # bright_sources_df.sort_values(by=['Peak_flux'], ascending=False, inplace=True) # Sorting from higher value to lower value of the peak flux

        # for index,row in bright_sources_df.iterrows():

        #     my_pointings = row['pointings']  # Get the pointings of each bright source
        #     pointings = set(my_pointings) 
    
        #     new_pointings = pointings - selfcal_pointings  # Exclude existing pointings

        #     if not new_pointings: 

        #         print('==> No pointings found to perform self-calibration')

        #         continue # this will skip to the next iteration of the index,row in bright_sources_df.iterrows() loop

        #     print('::: VOLS ::: ... The pointings ' + str(new_pointings) + ' are going to be self-calibrated')
    
        #     my_fields = ",".join(new_pointings)

        #     my_fields_str = ",".join(new_pointings) # TO BE WRITTEN LIKE 'P8,P9,P10'
        #     my_fields_join = "".join(new_pointings) # TO BE WRITTEN LIKE P8P9P10

        #     selfcal_pointings.update(new_pointings)  # Add new pointings to the set

        #     # centerbox[[x, y], [x_width, y_width]]

        #     my_peak_region = 'centerbox[[' + str(row['RA']) + 'deg,' + str(row['DEC']) + 'deg],[3arcsec,3arcsec]]'
        #     ra_shift = row['RA'] + (10/3600) 
        #     my_rms_region = 'centerbox[[' + str(ra_shift) + 'deg,' + str(row["DEC"]) + 'deg],[' + '20arcsec,20arcsec]]'

        #     print('::: VOLS ::: ... Starting cleaning and self-calibration process')

        #     print('::: VOLS ::: ... Splitting the visibility ' + my_vislist[i])

        #     os.system('rm -r ' + my_visFile+'.'+ my_submosaic + '.' +my_fields_join+ '.iter1')

        #     split(vis=my_visFile, 
        #       outputvis=my_visFile+'.'+ my_submosaic + '.' +my_fields_join+ '.iter1', # NOTE.-  NEED TO CHECK HOW THIS IS WRITTEN
        #       field=my_fields_str,
        #       datacolumn = 'data')
            
        #     print('::: VOLS ::: ... The measurement set ' + my_visFile+'.'+ my_submosaic + '.' +my_fields_join+ '.iter1 has been created')

        #     my_visFile_submosaic = my_visFile+'.'+ my_submosaic + '.' +my_fields_join+ '.iter1'

        #     print('::: VOLS ::: ... Creating individual images for each spectral window')

        #     my_spws = ['0','1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']

        #     for my_spw in my_spws:

        #         print('::: VOLS ::: ... Processing spectral window ' + my_spw)

        #         print('::: VOLS ::: ... Creating the dirty image')

        #         os.system('mkdir -p ' + my_dir+ 'images/each_spw/dirty')

        #         my_imageFile =  my_dir + 'images/each_spw/dirty/VOLS_dirty_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)+ '_'+ my_fields_join + '_spw' + my_spw

        #         os.system('rm -r ' + my_imageFile + '.*')

        #         delmod(vis=my_visFile_submosaic, otf=False)

        #         tclean(vis = my_visFile_submosaic,
        #                 uvrange = my_submosaicData['my_uvrange'][my_submosaic], # we want to create a mask
        #                 datacolumn = 'data',
        #                 spw = my_spw,
        #                 field = my_fields_str, # NOTE.- Actually, it would not be necessary since in the measurement set there are only those pointings
        #                 phasecenter = '',
        #                 imagename = my_imageFile,
        #                 imsize = [8250,8250],
        #                 cell = ['0.125arcsec'],
        #                 stokes = 'I',
        #                 specmode = 'mfs',
        #                 gridder = 'mosaic',
        #                 mosweight = False,
        #                 usepointing = False,
        #                 pblimit = 0.1,
        #                 deconvolver = 'hogbom',
        #                 #nterms = 2,
        #                 restoration = True,
        #                 pbcor = False,
        #                 weighting = 'briggs',
        #                 robust = 0.5,
        #                 npixels = 0,
        #                 niter = 0, # to make the dirty image
        #                 interactive = False,
        #                 restart = False,
        #                 savemodel = 'none',
        #                 calcres = True,
        #                 calcpsf = True,
        #                 parallel = False,
        #                 pbmask = 0.0,
        #                 )
                
        #         os.system('cp -r ' + my_imageFile + '.image .')

        #         print('::: VOLS ::: ... Checking the RMS from an emission-free region of the image, near the bright source')

        #         rms_stat = imstat(imagename = my_imageFile + '.image',region=my_rms_region)
        #         snr_stat = imstat(imagename= my_imageFile +'.image', region=my_peak_region)

        #         if 'rms' in rms_stat and len(rms_stat['rms']) > 0 and 'max' in snr_stat and len(snr_stat['max']) > 0:
        #             rms = rms_stat['rms'][0]  # better to check also if there is data in rms, to define another region, just in case the region we are measuring is empty but the source is on it                                                                                          
        #             peak = snr_stat['max'][0]
        #         else:
        #             print('Skipping iteration: Empty region found for ' + my_imageFile)
        #             continue

        #         print('::: VOLS ::: ... Calculating the SNR with a bright source in the region')

        #         snr = peak/rms

        #         print('==> rms for the submosaic using the pointings ' + my_fields_str + ': ' + str(rms))
        #         print('==> SNR for the submosaic using the pointings ' + my_fields_str + ': ' + str(snr))

        #         if rms > 5e-4:

        #             print('WARNING !!  The rms is larger than expected. Check you results carefully !')

        #         print('::: VOLS ::: ... Creating the dirty mask for spw ' + my_spw)

        #         print('::: VOLS ::: ... Setting a threshold of 20 sigma')

        #         threshold = 20 * rms

        #         print('==> threshold: '+ str(threshold) + ' Jy/beam')

        #         os.system('mkdir -p ' + my_dir + 'masks/each_spw/dirty')

        #         my_maskFile = my_dir + 'masks/each_spw/dirty/VOLS_dirty_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw

        #         os.system('rm -rf ' + my_maskFile + '.*')

        #         immath(
        #             imagename='VOLS_dirty_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw + '.image',
        #             expr='iif(IM0 >' + str(threshold) + ', 1.0, 0.0)',
        #             outfile=my_maskFile + '.mask'
        #             )

        #         os.system('rm -r VOLS_dirty_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw + '.image')

        #         print('::: VOLS ::: ... Creating a clean image using the dirty mask for spectral window '+ my_spw)

        #         os.system('mkdir -p ' + my_dir+ 'images/each_spw/clean')

        #         my_imageFile =  my_dir + 'images/each_spw/clean/VOLS_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)+ '_'+ my_fields_join + '_spw' + my_spw
        #         my_maskFile = my_dir + 'masks/each_spw/dirty/VOLS_dirty_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw

        #         os.system('rm -r ' + my_imageFile + '.*')

        #         delmod(vis=my_visFile_submosaic, otf=False)

        #         tclean(vis = my_visFile_submosaic,
        #                 uvrange = my_submosaicData['my_uvrange'][my_submosaic], # to create a mask
        #                 datacolumn = 'data',
        #                 spw = my_spw,
        #                 field = my_fields_str,
        #                 phasecenter = '',
        #                 imagename = my_imageFile,
        #                 imsize = [8250,8250],
        #                 cell = ['0.125arcsec'],
        #                 stokes = 'I',
        #                 specmode = 'mfs',
        #                 gridder = 'mosaic',
        #                 mosweight = False,
        #                 usepointing = False,
        #                 pblimit = 0.1,
        #                 deconvolver = 'hogbom',
        #                 #nterms = 2,
        #                 restoration = True,
        #                 pbcor = False,
        #                 weighting = 'briggs',
        #                 robust = 0.5,
        #                 npixels = 0,
        #                 niter = 1000000,
        #                 usemask = 'user',
        #                 mask = my_maskFile +'.mask',
        #                 threshold = '0.1mJy',
        #                 interactive = False,
        #                 restart = False,
        #                 savemodel = 'modelcolumn',
        #                 calcres = True,
        #                 calcpsf = True,
        #                 parallel = False,
        #                 pbmask = 0.0,
        #                 )
                
        #         os.system('cp -r ' + my_imageFile + '.image .')

        #         print('::: VOLS ::: ... Checking the RMS from an emission-free region of the image, near the bright source')

        #         rms_stat = imstat(imagename = my_imageFile + '.image',region=my_rms_region)
        #         snr_stat = imstat(imagename= my_imageFile +'.image', region=my_peak_region)

        #         rms = rms_stat['rms'][0]
        #         peak = snr_stat['max'][0]

        #         print('::: VOLS ::: ... Calculating the SNR with a bright source in the region')

        #         snr = peak/rms

        #         print('==> rms for the submosaic using the pointings ' + my_fields_str + ': ' + str(rms))
        #         print('==> SNR for the submosaic using the pointings ' + my_fields_str + ': ' + str(snr))

        #         if rms > 5e-4:

        #             print('WARNING !!  The rms is larger than expected. Check you results carefully !')

        #         print('::: VOLS ::: ... Creating the clean mask for spw ' + my_spw)

        #         print('::: VOLS ::: ... Setting a threshold of 20 sigma')

        #         threshold = 20 * rms

        #         print('==> Threshold: '+ str(threshold) + ' Jy/beam')

        #         os.system('mkdir -p ' + my_dir + 'masks/each_spw/clean')

        #         my_maskFile = my_dir + 'masks/each_spw/clean/VOLS_clean_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw

        #         os.system('rm -rf ' + my_maskFile + '.*')

        #         immath(
        #             imagename='VOLS_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw + '.image',
        #             expr='iif(IM0 >' + str(threshold) + ', 1.0, 0.0)',
        #             outfile=my_maskFile + '.mask'
        #             )

        #         os.system('rm -r VOLS_clean_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw + '.image')

        #         print('::: VOLS ::: ... Starting self-calibration in phase for spw ' + my_spw)

        #         prev_rms = rms # initiallizing rms
        #         prev_snr = snr # initiallizing snr

        #         os.system('mkdir -p ' + my_dir + 'calibration-tables')

        #         my_caltable = my_dir+'calibration-tables/caltable_'+str(my_dates[i])+'_'+ str(my_submosaic) +'_'+ my_fields_join + '_'+ my_calmode + '_spw' + my_spw+'.tb'

        #         print("::: VOLS ::: ... gaincal for self-calibration")

        #         gaincal(
        #             vis=my_visFile_submosaic,
        #             caltable=my_caltable,
        #             uvrange='',  # no restrictions in the uvrange
        #             gaintype='G',
        #             calmode=my_calmode,
        #             refant='ea10,ea23,ea28',
        #             minsnr=4,
        #             refantmode='strict',
        #             solint='inf',
        #             field=my_fields_str,
        #             spw=my_spw,
        #             )
                
        #         print("::: VOLS ::: ... applying calibration")

        #         applycal(
        #             vis=my_visFile_submosaic,
        #             uvrange='',
        #             gaintable=my_caltable,
        #             interp='linear',
        #             applymode='calonly',
        #             field=my_fields_str,
        #             )
                
        #         print('::: VOLS ::: ... Creating a self-calibrated image for spectral window '+ my_spw)

        #         os.system('mkdir -p ' + my_dir + 'images/each_spw/selfcal')

        #         # By default, all final images will be done excluding the short baselines (>35klambda)

        #         my_imageFile = my_dir + 'images/each_spw/selfcal/VOLS_selfcal_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) +'_'+ my_fields_join + '_'+ my_calmode + '_spw' + my_spw

        #         my_maskFile = my_dir + 'masks/each_spw/clean/VOLS_clean_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) + '_'+ my_fields_join + '_spw' + my_spw

        #         os.system('rm -rf ' + my_imageFile + '.*')

        #         tclean(vis=my_visFile_submosaic,
        #            uvrange='>35klambda', # final image, no need to create a mask
        #            datacolumn='corrected',
        #            spw=my_spw,
        #            field=my_fields_str,
        #            phasecenter='',
        #            imagename=my_imageFile,
        #            imsize=[8250,8250],
        #            cell=['0.125arcsec'],
        #            stokes='I',
        #            specmode='mfs',
        #            gridder='mosaic',
        #            mosweight=False,
        #            usepointing=False,
        #            pblimit=0.1,
        #            deconvolver='hogbom',
        #            #nterms=2,
        #            restoration=True,
        #            pbcor=False,
        #            weighting='briggs',
        #            robust=0.5,
        #            npixels=0,
        #            niter=10000,
        #            usemask='user',
        #            mask=my_maskFile + '.mask',
        #            threshold='0.1mJy',
        #            interactive=False,
        #            restart=False,
        #            savemodel='none',
        #            calcres=True,
        #            calcpsf=True,
        #            parallel=False,
        #            pbmask=0.0,
        #            )
            

        #         print('::: VOLS ::: ... Checking the RMS from an emission-free region of the image, near the bright source')

        #         rms_stat = imstat(imagename = my_imageFile + '.image',region=my_rms_region)
        #         snr_stat = imstat(imagename= my_imageFile +'.image', region=my_peak_region)


        #         rms = rms_stat['rms'][0]
        #         peak = snr_stat['max'][0]

        #         print('::: VOLS ::: ... Calculating the SNR with a bright source in the region')

        #         snr = peak/rms


        #         print('==> Current rms: ' + str(rms) + ' (selfcal)')
        #         print('==> Previous rms: ' + str(prev_rms) + ' (clean)')
 
        #         print('==> Current SNR: ' + str(snr) + ' (selfcal)')
        #         print('==> Previous SNR: ' + str(prev_snr) + '(clean)')
   
        #         if rms > 5e-4:
        #             print("WARNING !!  The rms is larger than expected. Check you images carefully !")
            
        #     print('::: VOLS ::: ... Each spectral window has been calibrated individually for the pointings ' +  my_fields_str + '. You can check the images now')
        
        # print('::: VOLS ::: ... Self-calibration of the brightest sources is finished')

        # print('::: VOLS ::: ... Pointings self-calibrated: ' + str(selfcal_pointings)) # not sure how is this variable written

        # # NOTE.- The self-calibrated measurement sets are written like 22A-195.sb41668223.eb41752682.59672.87566575232_cont.ms.(POINTINGSSELFCAL)

        # not_selfcal_pointings = all_pointings - selfcal_pointings
        # not_selfcal_pointings_str = ",".join(not_selfcal_pointings)

        # print('::: VOLS ::: ... Pointings not self-calibrated: ' + str(not_selfcal_pointings_str)) # not sure how is this variable written

        my_visFile_NOselfcal = my_visFile+ '.' + my_submosaic + '.NOselfcal.iter1'

        # os.system('rm -r ' + my_visFile_NOselfcal)

        # split(vis=my_visFile, 
        #       outputvis=my_visFile_NOselfcal, # NOTE.-  NEED TO CHECK HOW THIS IS WRITTEN
        #       field=not_selfcal_pointings_str,
        #       datacolumn = 'data')
              
        
        # points_to_concat = glob.glob(my_visFile + '.' + my_submosaic + "*.iter1") # NOTE.- This way, we concat the measurement sets of the submosaic (NOselfcal and with the self-cal pointings)
        #                                                                            # NOTE. This will ony work if we do not have measurement sets from iter 2 in the directory...

        # print('::: VOLS ::: ... Measurement sets to concatenate: ' + str(points_to_concat))

        
        # os.system('rm -r ' + my_visFile + '.' + my_submosaic+'.SELFCAL.BRIGHT_SOURCES')

        # concat(vis = points_to_concat, concatvis = my_visFile + '.' + my_submosaic+'.SELFCAL.BRIGHT_SOURCES')

        print('::: VOLS ::: ... Measurement set ' + my_visFile + '.' + my_submosaic+'.SELFCAL.BRIGHT_SOURCES has been created')

        print('NOTE.- This measurement sets contains the self-calibrated pointings that contain bright sources AND the ones that have not been self-calibrated')

        my_visFile_selfcal =  my_visFile + '.' + my_submosaic+'.SELFCAL.BRIGHT_SOURCES'

        # os.system('mkdir -p ' + my_dir + 'images/selfcal/bright_sources')

        my_imageFile = my_dir + 'images/selfcal/bright_sources/VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)
        my_maskFile = my_dir + 'masks/clean/VOLS_clean_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) 

        # print('::: VOLS ::: ... Creating the self-calibrated image with the calibration spw by spw on the pointings that contain bright sources')

        # os.system('rm -rf ' + my_imageFile + '.*')

        my_spws = '2~31' 

        # tclean(vis=my_visFile_selfcal,
        #            uvrange=my_submosaicData['my_uvrange'][my_submosaic], # >100kl because we are using this image to create the mask
        #            datacolumn='corrected',
        #            spw=my_spws,
        #            field=my_submosaicData['my_submosaicPointings'][my_submosaic],
        #            phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
        #            imagename=my_imageFile,
        #            imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
        #            cell=['0.125arcsec'],
        #            stokes='I',
        #            specmode='mfs',
        #            gridder='mosaic',
        #            mosweight=False,
        #            usepointing=False,
        #            pblimit=0.1,
        #            deconvolver='mtmfs',
        #            nterms=2,
        #            restoration=True,
        #            pbcor=False,
        #            weighting='briggs',
        #            robust=0.5,
        #            npixels=0,
        #            niter=10000,
        #            usemask='user',
        #            mask=my_maskFile + '.mask',  # using now the mask created with the clean image
        #            threshold='0.1mJy',
        #            interactive=False,
        #            restart=False,
        #            savemodel='none',
        #            calcres=True,
        #            calcpsf=True,
        #            parallel=False,
        #            pbmask=0.0,
        #            )
        
        # if my_submosaic == '01':

        #     my_imageFile = my_dir + 'images/selfcal/bright_sources/VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)+'_35klambda'
        #     my_maskFile = my_dir + 'masks/clean/VOLS_clean_mask_20sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) 

        #     print('::: VOLS ::: ... Creating another self-calibrated image excluding only short baselines (>35klambda)')

        #     os.system('rm -rf ' + my_imageFile + '.*')

        #     my_spws = '2~31' 

        #     tclean(vis=my_visFile_selfcal,
        #            uvrange='>35klambda', # >100kl because we are using this image to create the mask
        #            datacolumn='corrected',
        #            spw=my_spws,
        #            field=my_submosaicData['my_submosaicPointings'][my_submosaic],
        #            phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
        #            imagename=my_imageFile,
        #            imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
        #            cell=['0.125arcsec'],
        #            stokes='I',
        #            specmode='mfs',
        #            gridder='mosaic',
        #            mosweight=False,
        #            usepointing=False,
        #            pblimit=0.1,
        #            deconvolver='mtmfs',
        #            nterms=2,
        #            restoration=True,
        #            pbcor=False,
        #            weighting='briggs',
        #            robust=0.5,
        #            npixels=0,
        #            niter=10000,
        #            usemask='user',
        #            mask=my_maskFile + '.mask',  # using now the mask created with the clean image
        #            threshold='0.1mJy',
        #            interactive=False,
        #            restart=False,
        #            savemodel='none',
        #            calcres=True,
        #            calcpsf=True,
        #            parallel=False,
        #            pbmask=0.0,
        #            )


        # my_imageFile = my_dir + 'images/selfcal/bright_sources/VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)

        # os.system('cp -r ' + my_imageFile + '.image.tt0 .')

        # print('::: VOLS ::: ... Measuring the Median Absolute Deviation (MAD)')

        # selfcal_bright_sources_stats = imstat(imagename= my_imageFile +'.image.tt0')
        # mad = selfcal_bright_sources_stats['medabsdevmed'][0]

        # print('==> MAD: '+ str(mad) + ' Jy/beam')

        # print('::: VOLS ::: ... Setting a threshold of MAD x 1.5 x 10, similar to a 10sigma threshold')

        # threshold = mad*1.5*10

        # print('::: VOLS ::: ... Creating the mask for submosaic ' + str(my_submosaic) + ' using the self-calibrated image')

        # os.system('mkdir -p ' + my_dir + 'masks/selfcal')

        # my_maskFile = my_dir + 'masks/selfcal/VOLS_selfcal_mask_10sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)

        # os.system('rm -rf ' + my_maskFile + '.*')

        # # NOTE.- not sure why but immath is not working if the image is not in the same directory.

        # immath(
        #             imagename='VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)  + '.image.tt0',
        #             expr='iif(IM0 >' + str(threshold) + ', 1.0, 0.0)',
        #             outfile=my_maskFile + '.mask'
        #         )

        # os.system('rm -r VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)  + '.image.tt0')

        # os.system('mkdir -p ' + my_dir + 'images/selfcal/bright_sources')

        # my_imageFile = my_dir + 'images/selfcal/bright_sources/VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)+'_hogbom'
        # my_imageName = 'VOLS_selfcal_bright_sources_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic)+'_hogbom'
        # my_maskFile = my_dir + 'masks/selfcal/VOLS_selfcal_mask_10sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) 

        # print('::: VOLS ::: ... Creating the self-calibrated image using hogbom and the mask at 10sigma')

        # os.system('rm -rf ' + my_imageFile + '.*')


        # tclean(vis=my_visFile_selfcal,
        #            uvrange='>35klambda', # >100kl because we are using this image to create the mask
        #            datacolumn='corrected',
        #            spw=my_spws,
        #            field=my_submosaicData['my_submosaicPointings'][my_submosaic],
        #            phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
        #            imagename=my_imageFile,
        #            imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
        #            cell=['0.125arcsec'],
        #            stokes='I',
        #            specmode='mfs',
        #            gridder='mosaic',
        #            mosweight=False,
        #            usepointing=False,
        #            pblimit=0.1,
        #            deconvolver='hogbom',
        #            #nterms=2,
        #            restoration=True,
        #            pbcor=False,
        #            weighting='briggs',
        #            robust=0.5,
        #            npixels=0,
        #            niter=10000,
        #            usemask='user',
        #            mask=my_maskFile + '.mask',  # using now the mask created with the clean image
        #            threshold='0.1mJy',
        #            interactive=False,
        #            restart=False,
        #            savemodel='modelcolumn',
        #            calcres=True,
        #            calcpsf=True,
        #            parallel=False,
        #            pbmask=0.0,
        #            )
        
        # exportfits(imagename = my_imageFile + '.image', fitsimage = my_imageFile + '.fits', overwrite = True)
        
        # print('::: VOLS ::: ... Saving the model')

        # print('::: VOLS ::: ... Splitting the pointings already self-calibrated')

        # selfcal_pointings = {'P10', 'P8', 'P9'} # REMOVE THIS AFTER USING IT

        # my_selfcal_pointings_str = ",".join(selfcal_pointings)

        # os.system('rm -r ' + my_visFile + '.' + my_submosaic+'.SELFCAL.iter2')

        # split(vis = my_visFile_selfcal,
        #         outputvis = my_visFile + '.' + my_submosaic+'.SELFCAL.iter2', # This dataset contains ONLY the pointings that are already self-calibrated (spw by spw, during the first iteration)
        #         field = my_selfcal_pointings_str,
        #         datacolumn = 'corrected')
        

        # print('::: VOLS ::: ... Running pyBDSF on the self-calibrated image')

        # my_imageFits = my_imageFile + '.fits' 

        # print('==> Processing '+ my_imageFits)

        # # Run PyBDSF with the specified parameters

        # img = bdsf.process_image(my_imageFits,
        #                      adaptive_rms_box=True,
        #                      thresh_isl=10.0,
        #                      thresh_pix=10.0)
    

        # my_catalog = my_dir + 'data/' + my_imageName  + '_10sigma_catalog.csv'
        
        # os.system('rm -r ' + my_catalog)

        # # Write the source catalog

        # img.write_catalog(outfile=my_catalog,
        #             catalog_type = 'srl',
        #             format = 'csv') 

        # print('::: VOLS ::: ... Finishing pyBDSF')

        # print('==> You can check the catalog at 10sigma in ' + my_dir + 'data/' + my_imageName + '_10sigma_catalog.csv')

        # column_names = [ "Source_id", "Isl_id", "RA", "E_RA", "DEC", "E_DEC", "Total_flux", "E_Total_flux", "Peak_flux", 
        #                 "E_Peak_flux", "RA_max", "E_RA_max", "DEC_max", "E_DEC_max", "Maj", "E_Maj","Min", "E_Min", "PA", 
        #                 "E_PA", "Maj_img_plane", "E_Maj_img_plane", "Min_img_plane","E_Min_img_plane", "PA_img_plane", 
        #                 "E_PA_img_plane", "DC_Maj", "E_DC_Maj", "DC_Min","E_DC_Min", "DC_PA", "E_DC_PA", "DC_Maj_img_plane", 
        #                 "E_DC_Maj_img_plane", "DC_Min_img_plane","E_DC_Min_img_plane", "DC_PA_img_plane", "E_DC_PA_img_plane", 
        #                 "Isl_Total_flux", "E_Isl_Total_flux", "Isl_rms", "Isl_mean", "Resid_Isl_rms", "Resid_Isl_mean", "S_Code"
        #                 ]
        
        # my_catalog_df = pd.read_csv(my_catalog, comment='#', names=column_names)

        # coords_deg = SkyCoord(ra=my_catalog_df['RA'].values * u.degree, dec=my_catalog_df['DEC'].values * u.degree, frame='icrs')

        # my_catalog_df['RA_hour_angle'] = coords_deg.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad = True)
        # my_catalog_df['DEC_degree'] = coords_deg.dec.to_string(unit=u.degree, sep=':', precision=2, pad=True)
    

        # error_coords = SkyCoord(ra=my_catalog_df['E_RA'].values * u.degree, dec=my_catalog_df['E_DEC'].values * u.degree, frame='icrs')
        # my_catalog_df['E_RA_hour_angle'] = error_coords.ra.to_string(unit=u.hourangle, sep=':', precision=4, pad=True)
        # my_catalog_df['E_DEC_degree'] = error_coords.dec.to_string(unit=u.degree, sep=':', precision=4, pad=True)

        # print('::: VOLS ::: ... Checking the pointings that contain each source')

        # listobs = pd.read_csv('vols-listobs-cband.csv', sep=';', header=0) # NOTE.- NEED TO ADD THIS TO EACH SERVER

        # coords = SkyCoord(ra=listobs['RA'], dec=listobs['Decl'], unit=(u.hourangle, u.deg))

        # my_catalog_df['pointings'] = [[] for _ in range(len(my_catalog_df))]

        # for index, row in my_catalog_df.iterrows():
    
        #     my_source = SkyCoord(row['RA_hour_angle'], row['DEC_degree'], unit=(u.hourangle, u.deg))

        #     points_in_region = []
            
        #     for k, coord in enumerate(coords):
        #         # Calculate the separation between the current pointing and the target
        #         separation = coord.separation(my_source)

        #         # If the separation is within the beam radius, add to points_in_region
        #         if separation <= beam_radius * u.deg:
        #             points_in_region.append(listobs.iloc[k]['Name']) # storing as a list

        #     my_catalog_df.at[index, 'pointings'] = points_in_region 

        # my_catalog_df.sort_values(by=['Peak_flux'], ascending=False, inplace=True) # Sorting from higher value to lower value of the peak flux

        # my_catalog_df.to_csv(my_catalog, index=False)

        # for index,row in my_catalog_df.iterrows():

        #     my_pointings = row['pointings']  # Get the pointings of each bright source
        #     #pointings = set(my_pointings.split(",")) 
        #     pointings = set(my_pointings) 
    
        #     new_pointings = pointings - selfcal_pointings  # Exclude existing pointings

        #     if not new_pointings: 

        #         print('==> No pointings found to perform self-calibration')

        #         continue # this will skip to the next iteration of the index,row in bright_sources_df.iterrows() loop

        #     print('::: VOLS ::: ... The pointings ' + str(new_pointings) + ' are going to be self-calibrated')
    
        #     my_fields = ",".join(new_pointings)

        #     my_fields_str = ",".join(new_pointings)# TO BE WRITTEN LIKE 'P8,P9,P10'
        #     my_fields_join = "".join(new_pointings) # TO BE WRITTEN LIKE P8P9P10

        #     selfcal_pointings.update(new_pointings)  # Add new pointings to the set

        #     print('::: VOLS ::: ... Splitting the visibility ' + my_vislist[i])

        #     os.system('rm -r ' + my_visFile+'.'+ my_submosaic + '.' +my_fields_join+'.iter2')

        #     split(vis=my_visFile, 
        #       outputvis=my_visFile+'.'+ my_submosaic + '.' +my_fields_join+'.iter2', # NOTE. This will ony
        #       field=my_fields_str,
        #       datacolumn = 'data')
            
        #     my_visFile_submosaic = my_visFile+'.'+ my_submosaic + '.' +my_fields_join+'.iter2'
            
        #     print('::: VOLS ::: ... The measurement set ' + my_visFile+'.'+ my_submosaic + '.' +my_fields_join+'.iter2 has been created')

        #     print('::: VOLS ::: ... Starting self-calibration in phase using the model with all the spws')

        #     my_caltable = my_dir+'calibration-tables/caltable_'+str(my_dates[i])+'_'+ str(my_submosaic) +'_'+ my_fields_join + '_'+ my_calmode + '.tb'

        #     print("::: VOLS ::: ... gaincal for self-calibration")

        #     gaincal(
        #             vis=my_visFile_submosaic,
        #             caltable=my_caltable,
        #             uvrange='',  # no restrictions in the uvrange
        #             gaintype='G',
        #             calmode=my_calmode,
        #             refant='ea10,ea23,ea28',
        #             minsnr=4,
        #             refantmode='strict',
        #             solint='inf',
        #             field=my_fields_str,
        #             spw=my_spws,
        #             )
                
        #     print("::: VOLS ::: ... applying calibration")

        #     applycal(
        #             vis=my_visFile_submosaic,
        #             uvrange='',
        #             gaintable=my_caltable,
        #             interp='linear',
        #             applymode='calonly',
        #             field=my_fields_str,
        #             )
        
        # not_selfcal_pointings = all_pointings - selfcal_pointings
        # not_selfcal_pointings_str = ",".join(not_selfcal_pointings)

        # my_visFile_NOselfcal_final = my_visFile+ '.' + my_submosaic + '.NOselfcal.iter2'

        # os.system('rm -r ' + my_visFile_NOselfcal_final)

        # split(vis=my_visFile_NOselfcal, 
        #       outputvis=my_visFile_NOselfcal_final, # NOTE.-  NEED TO CHECK HOW THIS IS WRITTEN
        #       field=not_selfcal_pointings_str,
        #       datacolumn = 'data')
        
        # points_to_concat = glob.glob(my_visFile + '.' + my_submosaic + "*.iter2") # NOTE.- This way, we concat the measurement sets of the submosaics that have been self-calibrated in the second round 

        # print('::: VOLS ::: ... Measurement sets to concatenate: ' + str(points_to_concat))

        my_visFile_final = my_visFile + '.' + my_submosaic+'.SELFCAL.FINAL' # NO! we have to concatenate also the self-calibrated points!!

        # os.system('rm -r '+ my_visFile_final)

        # concat(vis = points_to_concat, concatvis = my_visFile_final)

        my_imageFile = my_dir + 'images/selfcal/final/VOLS_selfcal_final_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) +'_'+ my_calmode
        my_maskFile = my_dir + 'masks/selfcal/VOLS_selfcal_mask_10sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) 

        print('::: VOLS ::: ... Creating the self-calibrated image after calibrating the pointings with weak sources')

        os.system('rm -rf ' + my_imageFile + '.*')	

        tclean(vis=my_visFile_final,
                   uvrange='>35klambda', 
                   datacolumn='corrected',
                   spw=my_spws,
                   field=my_submosaicData['my_submosaicPointings'][my_submosaic],
                   phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
                   imagename=my_imageFile,
                   imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
                   cell=['0.125arcsec'],
                   stokes='I',
                   specmode='mfs',
                   gridder='mosaic',
                   mosweight=False,
                   usepointing=False,
                   pblimit=0.1,
                   deconvolver='mtmfs',
                   nterms=2,
                   restoration=True,
                   pbcor=False,
                   weighting='briggs',
                   robust=0.5,
                   npixels=0,
                   niter=10000,
                   usemask='user',
                   mask=my_maskFile + '.mask',  
                   threshold='0.1mJy',
                   interactive=False,
                   restart=False,
                   savemodel='none',
                   calcres=True,
                   calcpsf=True,
                   parallel=False,
                   pbmask=0.0,
                   )
        
        if my_submosaic == '01':

            my_imageFile = my_dir + 'images/selfcal/final/VOLS_selfcal_final_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) +'_'+ my_calmode+'_allBaselines'
            my_maskFile = my_dir + 'masks/selfcal/VOLS_selfcal_mask_10sigma_Cband_cont_' + str(my_dates[i]) + '_' + str(my_submosaic) 

            print('::: VOLS ::: ... Creating and additional self-calibrated image using all baselines')

            os.system('rm -rf ' + my_imageFile + '.*')	

            tclean(vis=my_visFile_final,
                   uvrange='', 
                   datacolumn='corrected',
                   spw=my_spws,
                   field=my_submosaicData['my_submosaicPointings'][my_submosaic],
                   phasecenter=my_submosaicData['my_submosaicPhaseCenter'][my_submosaic],
                   imagename=my_imageFile,
                   imsize=my_submosaicData['my_submosaicImsize'][my_submosaic],
                   cell=['0.125arcsec'],
                   stokes='I',
                   specmode='mfs',
                   gridder='mosaic',
                   mosweight=False,
                   usepointing=False,
                   pblimit=0.1,
                   deconvolver='mtmfs',
                   nterms=2,
                   restoration=True,
                   pbcor=False,
                   weighting='briggs',
                   robust=0.5,
                   npixels=0,
                   niter=10000,
                   usemask='user',
                   mask=my_maskFile + '.mask',  
                   threshold='0.1mJy',
                   interactive=False,
                   restart=False,
                   savemodel='none',
                   calcres=True,
                   calcpsf=True,
                   parallel=False,
                   pbmask=0.0,
                   )
            
        #sys.stdout.close()
        #sys.stdout = original_stdout

print('==> The script is done, check (and enjoy) your images now')
