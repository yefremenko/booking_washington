import numpy as np
import pandas as pd

columns_to_transform = [
    'host_response_rate',
    'host_acceptance_rate',
    'cleaning_fee',
    'host_since'
                        ]

categorical_features = [
    'host_neighbourhood',
    'city',
    'state',
    'property_type',
    'room_type',
    'bed_type',
    'instant_bookable',
    'cancellation_policy'
                       ]

top_category_host_neighbourhood = [
    'Capitol Hill',
    'Columbia Heights',
    'Dupont Circle',
    'Logan Circle',
    'U Street Corridor',
    'Adams Morgan',
    'Near Northeast/H Street Corridor',
    'Shaw'
]

low_category_property_type = [
    'Condominium',
    'Townhouse',
    'Bed & Breakfast',
    'Loft',
    'Other',
    'Dorm',
    'Boat',
    'Cabin',
    'unknown',
    'Bungalow'
]

room_type_values = [
    'Private room',
    'Shared room'
]

cancellation_policy_values = [ 
    'moderate',
    'strict',
    'super_strict_30'
]

host_verifications_values = [
      'email',
      'facebook',
      'google',
      'jumio',
      'kba',
      'linkedin',
       'manual_offline',
      'manual_online',
      'phone',
      'reviews',
      'sent_id'
]

rare_amenities_name = [
    'Gym',
    'Petsliveonthisproperty',
    'Buzzer/WirelessIntercom',
    'IndoorFireplace',
    'SafetyCard',
    'PetsAllowed',
    'Doorman',
    'Dog(s)',
    'WheelchairAccessible',
    'Pool',
    'Breakfast',
    'Cat(s)',
    'SuitableforEvents',
    'HotTub',
    'SmokingAllowed',
    'Otherpet(s)',
    'Washer/Dryer'
]

midle_popular_amenities_name = [
    'Washer',
    'Dryer',
    'SmokeDetector',
    'Internet',
    'Essentials',
    'TV',
    'Shampoo',
    'CableTV',
    'FireExtinguisher',
    'Family/KidFriendly',
    'CarbonMonoxideDetector',
    'FirstAidKit',
    'FreeParkingonPremises',
    'ElevatorinBuilding'
]

amenities_values = [
    'Internet',
    'WirelessInternet',
    'AirConditioning',
    'Kitchen',
    'Buzzer/WirelessIntercom',
    'Heating',
    'Family/KidFriendly',
    'Washer',
    'Dryer',
    'SmokeDetector', 
    'CarbonMonoxideDetector',
    'FireExtinguisher',
    'Essentials', 
    'Shampoo',
    'TV',
    'CableTV',
    'Petsliveonthisproperty',
    'WheelchairAccessible',
    'PetsAllowed',
    'IndoorFireplace',
    'FreeParkingonPremises',
    'Doorman',
    'Gym',
    'ElevatorinBuilding',
    'FirstAidKit',
    'Dog(s)',
    'Breakfast',
    'HotTub',
    'SafetyCard',
    'Pool', 
    'Cat(s)',
    'SuitableforEvents',
    'SmokingAllowed',
    'Otherpet(s)',
    'Washer/Dryer'
]

train_features = [
    'host_since',
    'host_response_rate',
    'host_acceptance_rate',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'cleaning_fee',
    'guests_included',
    'minimum_nights',
    'maximum_nights',
    'number_of_reviews',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'calculated_host_listings_count',
    'reviews_per_month',
    'host_response_rate_nan',
    'host_acceptance_rate_nan',
    'Capitol Hill',
    'Columbia Heights',
    'Dupont Circle',
    'Logan Circle', 
    'U Street Corridor',
    'Adams Morgan',
    'Near Northeast/H Street Corridor',
    'Shaw', 
    'city_one_hot',
    'state_one_hot',
    'property_type_house',
    'property_type_other',
    'Private room',
    'Shared room',
    'bed_type_not_real_bed',
    'instant_bookable_not_f', 
    'moderate', 'strict',
    'super_strict_30',
    'email', 
    'facebook', 
    'google',
    'jumio',
    'kba', 
    'linkedin',
    'manual_offline',
    'manual_online',
    'phone',
    'reviews',
    'sent_id',
    'Washer_amenities',
    'Dryer_amenities',
    'SmokeDetector_amenities',
    'Internet_amenities',
    'Essentials_amenities',
    'TV_amenities',
    'Shampoo_amenities',
    'CableTV_amenities',
    'FireExtinguisher_amenities',
    'Family/KidFriendly_amenities',
    'CarbonMonoxideDetector_amenities',
    'FirstAidKit_amenities', 
    'FreeParkingonPremises_amenities',
    'ElevatorinBuilding_amenities',
    'other_rare_amenities_amenities'
]

def create_one_hot_feature_by_key(df_col,keys):
    one_hot = []
    for key in keys:
        one_hot_prt = df_col.str.contains(key)*1
        one_hot.append(one_hot_prt)
    return one_hot

def data_preprocess(df_train):

    df_train.loc[:,'host_since'] = pd.to_datetime(df_train['host_since'])
    end_time_point = np.datetime64('2016-01-01')
    df_train.loc[:, 'host_since'] = \
        (end_time_point - df_train.host_since).dt.days
    
    df_train.loc[:,'host_response_rate_nan'] = 0
    df_train.loc[df_train.host_response_rate.isnull(),'host_response_rate_nan'] = 1
    df_train.loc[:,'host_response_rate'] = df_train.host_response_rate.fillna('0%')
    df_train.loc[:,'host_response_rate'] = [int(i.replace("%","")) for i in df_train.host_response_rate]
    
    df_train.loc[:,'host_acceptance_rate_nan'] = 0
    df_train.loc[df_train.host_acceptance_rate.isnull(),'host_acceptance_rate_nan'] = 1
    df_train.loc[:,'host_acceptance_rate'] = df_train.host_acceptance_rate.fillna('0%')
    df_train.loc[:,'host_acceptance_rate'] = [int(i.replace("%","")) for i in df_train.host_acceptance_rate]

    df_train.loc[:,'cleaning_fee'] = df_train.cleaning_fee.fillna('0$')
    df_train.loc[:,'cleaning_fee'] = [float(i.replace("$","")) for i in df_train.cleaning_fee]
    
    df_train.loc[:,'host_neighbourhood'] = df_train.host_neighbourhood.fillna('unknown')
    host_neighbourhood_one_hot_top = pd.get_dummies(df_train.host_neighbourhood)[top_category_host_neighbourhood]
    df_train = pd.concat([df_train,host_neighbourhood_one_hot_top],axis = 1).drop(columns = 'host_neighbourhood')
    
    df_train.loc[~df_train['city'].str.lower().str.contains('washington'),'city_one_hot'] = 1
    df_train.loc[:,'city_one_hot'] = df_train.city_one_hot.fillna(0)
    df_train = df_train.drop(columns = 'city')
    
    df_train.loc[~df_train['state'].str.lower().str.contains('dc'),'state_one_hot'] = 1
    df_train.loc[:,'state_one_hot'] = df_train.state_one_hot.fillna(0)
    df_train = df_train.drop(columns = 'state')
    
    df_train.loc[:,'property_type'] = df_train.property_type.fillna('unknown')
    df_train.loc[df_train['property_type'] == 'House','property_type_house'] = 1
    df_train.loc[:,'property_type_house'] = df_train.property_type_house.fillna(0)
    
    df_train.loc[df_train['property_type'].isin(low_category_property_type),'property_type_other'] = 1
    df_train.loc[:,'property_type_other'] = df_train.property_type_other.fillna(0)
    df_train = df_train.drop(columns = 'property_type')
    
    room_type_one_hot = pd.get_dummies(df_train.room_type,drop_first=True)[room_type_values]
    df_train = pd.concat([df_train,room_type_one_hot],axis = 1).drop(columns = 'room_type')
    
    df_train.loc[df_train['bed_type'] != 'Real Bed','bed_type_not_real_bed'] = 1
    df_train.loc[:,'bed_type_not_real_bed'] = df_train.bed_type_not_real_bed.fillna(0)
    df_train = df_train.drop(columns = 'bed_type')
    
    df_train.loc[df_train['instant_bookable'] != 'f','instant_bookable_not_f'] = 1
    df_train.loc[:,'instant_bookable_not_f'] = df_train.instant_bookable_not_f.fillna(0)
    
    cancellation_policy_one_hot = pd.get_dummies(df_train.cancellation_policy)[cancellation_policy_values]
    df_train = pd.concat([df_train,cancellation_policy_one_hot],axis = 1).drop(columns = 'cancellation_policy')
    
    df_train.loc[:,'host_verifications'] = [i.replace('[','') for i in df_train.host_verifications]
    df_train.loc[:,'host_verifications'] = [i.replace(']','') for i in df_train.host_verifications]
    df_train.loc[:,'host_verifications'] = [i.replace("'",'') for i in df_train.host_verifications]
    df_train.loc[:,'host_verifications'] = [i.replace(" ",'') for i in df_train.host_verifications]
    host_verifications_one_hot = create_one_hot_feature_by_key(df_train.host_verifications,host_verifications_values)
    df_train = df_train.join(pd.concat(host_verifications_one_hot,keys=[value for value in host_verifications_values],axis = 1))
    df_train = df_train.drop(columns = ['host_verifications'])
    
    df_train.loc[:,'amenities'] = [i.replace('{','') for i in df_train.amenities]
    df_train.loc[:,'amenities'] = [i.replace('}','') for i in df_train.amenities]
    df_train.loc[:,'amenities'] = [i.replace("'",'') for i in df_train.amenities]
    df_train.loc[:,'amenities'] = [i.replace(" ",'') for i in df_train.amenities]
    df_train.loc[:,'amenities'] = [i.replace('"','') for i in df_train.amenities]
    amenities_one_hot = create_one_hot_feature_by_key(df_train.amenities,amenities_values)
    amenities_one_hot = pd.concat(amenities_one_hot,keys=[value for value in amenities_values],axis = 1)
    midle_popular_amenities_one_hot = amenities_one_hot[midle_popular_amenities_name]
    rare_amenities_one_hot = amenities_one_hot[rare_amenities_name].apply(lambda x: (x.sum() != 0)*1 ,axis = 'columns')
    midle_popular_amenities_one_hot.loc[:,'other_rare_amenities'] = rare_amenities_one_hot
    midle_popular_amenities_one_hot = midle_popular_amenities_one_hot.add_suffix('_amenities')
    df_train = df_train.join(midle_popular_amenities_one_hot)
    df_train = df_train.drop(columns = ['amenities'])
    final_train_df = df_train[train_features].fillna(0)
    return final_train_df
    