import numpy as np
import pandas as pd
import string
from collections import Counter
import requests
from difflib import SequenceMatcher
import collections
import math
import streamlit as st
import openpyxl

direction_dict={
        "S": "SOUTH",
        "w": "WEST",
        "E": "EAST",
        "N": "NORTH",
        "NW" : "NORTHWEST",
        "SW": "SOUTHWEST",
        "NE": "NORTHEAST",
        "SE": "SOUTHEAST"

}
appendix_dict = {
        "STREET":"ST" ,
        "AVENUE" :"AVE",
        "COURT":"CT",
        "BOULEVARD" :"BLVD",
        "SUITE" :"STE" ,
        "ALLEY" :"ALY" ,
        "GATEWAY":"GTWY" ,
        "HARBOR":"HBR" ,
        "HIGHWAY":"HWY" ,
        "PARKWAY":"PKWY" ,
        "PLACE":"PL" ,
        "BUILDING":"BLDG" ,
        "ROAD":"RD",
        "SAINT":"ST."
}
spec_char_dict = {
        "Â" : "A",
        "Á" : "A",
        "Ç" : "C",
        "É" : "E",
        "È" : "E",
        "Ï" : "I",
        "Í" : "I",
        "Ñ" : "N",
        "Ô" : "O",
        "Ó" : "O",
        "Ú" : "U",
        "Ü" : "U"
        
}
state_dict = {
    "ALABAMA":"AL",
    "ALASKA":"AK",
    "ARIZONA":"AZ",
    "ARKANSAS":"AK",
    "CALIFORNIA":"CA",
    "COLORADO":"CO",
    "CONNECTICUT":"CT",
    "DALAWARE":"DE",
    "FLORIDA":"FL",
    "GEORGIA":"GA",
    "HAWAII":"HI",
    "IDAHO":"ID",
    "ILLINOIS":"IL",
    "IOWA":"IA",
    "KANSAS":"KS",
    "KENTUCKY":"KY",
    "LOUISIANA":"LA",
    "MAINE":"ME",
    "DC":"MD",
    "MARYLAND":"MD",
    "MASSACHUSETTS":"MA",
    "MIVHIGAN":"MI",
    "MINNESOTA":"MN",
    "MISSISSIPPI":"MS",
    "MISSOURI":"MO",
    "MONTANA":"MT",
    "NEBRASKA":"NE",
    "NEVADA":"NV",
    "NEW HAMPSHIRE":"NH",
    "NEW JERSEY":"NJ",
    "NEW MEXICO":"NM",
    "NEW YORK":"NY",
    "NORTH CAROLINA":"NC",
    "NORTH DAKOTA":"ND",
    "OHIO":"OH",
    "OKLAHOMA":"OK",
    "OREGON":"OR",
    "PENNSYLVANIA":"PA",
    "RHODE ISLAND":"RI",
    "SOUTH CAROLINA":"SC",
    "SOUTH DAKOTA":"SD",
    "TENNESSEE":"TN",
    "TEXAS":"TX",
    "UTAH":"UT",
    "VERMONT":"VT",
    "VIRGINIA":"VA",
    "WASHINGTON":"WA",
    "WEST VIRGINIA":"WV",
    "WISCONSIN":"WI",
    "WYOMING":"WY",

    "ONTARIO":"ON",
    "QUEBEC":"QC",
    "BRITISH COLUMBIA":"BC",
    "ALBERTA":"AB",
    "MANITOBA":"MB",
    "SASKATCHEWAN":"SK",
    "NOVA SCOTIA":"NS",
    "NEW BRUNSWICK":"NB",
    "NEWFOUNDLAND AND LABRADOR":"NL",
    "PRINCE EDWARD ISLAND":"PE"


}

country_dict = {
        "USA" : "US",
        "CANADA" : "CA",
        "CAN" : "CA",
        "UNITED STATES" :"US",
        "MEXICO" : "MX",
        "MEX":"MX",
        "PUERTO RICO" :"US",
        "PR":"US",
        "PRI":"US",
        "COSTA RICA":"CR",
        "ARGENTINA":"AR",
        "AUSTRALIA":"AU",
        "BANGLADESH":"BD",
        "BELGIUM":"BE",
        "BRAZIL":"BR",
        "SWIZERLAND":"CH",
        "GERMANY":"DE",
        "DENMARK":"DK",
        "SPAIN":"ES",
        "FINLAND":"FI",
        "FRANCE":"FR",
        "GREAT BRITAIN":"GB",
        "UNITED KINGDOM":"GB",
        "UK":"GB",
        "INDIA":"IN",
        "ITALY":"IT",
        "JAPAN":"JP",
        "MALAYSIA":"MY",
        "PHILLIPPINES":"PH",
        "RUSSIA":"RU",
        "THAILAND":"TH"
}

def remove_punctuation(input_string):
    input_string = str(input_string)
    translator = str.maketrans('','',string.punctuation)
    cleaned_string = input_string.translate(translator)
    return cleaned_string

def replace_spec_char(input_string):
    for special_char, replacement in spec_char_dict.items():
        input_string = input_string.replace(special_char,replacement)
    return input_string

def state_corrector(state):
    statedf = state.copy()
    correct_state = []
    for astate in statedf:
        clean_state = replace_spec_char(remove_punctuation(astate).upper())
        if clean_state in state_dict:
            correct_state.append(state_dict[clean_state])
        else:
            correct_state.append(clean_state)
    return correct_state

def country_corrector(country):
    countrydf = country.copy()
    correct_country = []
    for acountry in countrydf:
        clean_country = replace_spec_char(remove_punctuation(acountry).upper())
        if clean_country in country_dict:
            correct_country.append(country_dict[clean_country])
        else:
            correct_country.append(clean_country)
    return correct_country

def city_corrector(city):
    citydf = city.copy()
    correct_city = []
    for acity in citydf:
        clean_city = replace_spec_char(remove_punctuation(acity).upper())
        correct_city.append(clean_city)
    return correct_city

def get_location_info(zipcode, std_country):
    if std_country == "CA":
        zipcode = zipcode[:3]
    else:
        pass
    base_url = f"http://api.zippopotam.us/{std_country}/{zipcode}"
    
    response = requests.get(base_url)
    data = response.json()

    if 'places' in data and data['places']:  # Check if 'places' key exists and has data
        place = data['places'][0]
        city = place.get('place name', '').upper()
        state = place.get('state', '').upper()
    else:
        city = None
        state = None

    if state in state_dict:
        c_state = state_dict[state]
    else:
        c_state = state
    

    return city, c_state

def correct_n_zip(country,mixed_list):
    correct_zips = []
    zipcodes =[]
    for item in mixed_list:
        if isinstance(item, float) and math.isnan(item):
            continue  # Skip NaN values
        czip = str(int(item)) if isinstance(item, float) else str(item)
        zipcodes.append(czip)
    for acountry,azip in zip(country,zipcodes):
        if acountry in ["DE","US","ES","FI","FR","IT","MX","MY","TH","AR","AU","BD","BE","CH","DK","PH","IN","RU","CA","GB"]:
            n = 5
            if acountry in ["AR","AU","BD","BE","CH","DK","PH"]:
                n = 4
            elif acountry in ["IN","RU","CA"]:
                n = 6
            elif acountry in ["GB"]:
                n = 3
            azip = str(azip)
            azip = remove_punctuation(azip).replace(" ", "")
            zip_length = len(azip)
            if zip_length < n:
                correct_zips.append(str((n - zip_length) * '0' + azip))
            else:
                correct_zips.append(str(azip[:n]))
        else:
            correct_zips.append(str(azip))
    return correct_zips

def address_corrector(addresses):

    addressdf = addresses.copy()
    correct_address = []
    for aaddress in addressdf:
        aaddress_cleaned = replace_spec_char(remove_punctuation(aaddress).upper())
        add_word = aaddress_cleaned.split()
        result_address = []
        for aword in add_word:
            if aword in direction_dict:
                result_address.append(direction_dict[aword])
            elif aword in appendix_dict:
                result_address.append(appendix_dict[aword])
            else:
                result_address.append(aword)
        std_add = " ".join(result_address)
        correct_address.append(std_add)
            
    return correct_address

def normalize_address(address):
    return ' '.join(sorted(address.split()))


def replace_diffseq_addresses(addresses):
    normalized_addresses = [normalize_address(address) for address in addresses]
    address_counts = Counter(normalized_addresses)
    result = []
    for address in addresses:
        normalized_address = normalize_address(address)
        if address_counts[normalized_address] > 1:
            most_common = max(
                (addr for addr in addresses if normalize_address(addr) == normalized_address),
                key=addresses.count
            )
            result.append(most_common)
        else:
            result.append(address)
    
    return result

def get_similarity(str1,str2):
    return SequenceMatcher(None,str1,str2).ratio()

class UnionFind:
    def __init__(self,nodes: list) -> None:
        self.fathers = {n:n for n in nodes}
        self.counts = {n:[n]for n in nodes}     
        # self.sizes = {n:1 for n in nodes}

    def find(self,node):
        father = self.fathers[node]
        if father != node:

            # if father != self.fathers[father]:
            #     self.sizes[father] -= 1

            father = self.find(father)
        
        self.fathers[node] = father
        return father

    def union(self,n1,n2):
        n1_father = self.find(n1)
        n2_father = self.find(n2)

        if len(self.counts[n1_father]) >= len(self.counts[n2_father]):
            self.fathers[n2_father] = n1_father
            # self.sizes[n1_father] = self.sizes[n1_father] + self.sizes[n2_father]
            self.counts[n1_father] += self.counts[n2_father]
            self.counts[n2_father] = []
        else:
            self.fathers[n1_father] = n2_father
            # self.sizes[n2_father] = self.sizes[n1_father] + self.sizes[n2_father]
            self.counts[n2_father] += self.counts[n1_father]
            self.counts[n1_father] = []


def get_similarity(str1,str2):
    return SequenceMatcher(None,str1,str2).ratio()    


def preprocess(location_list):

    grouped_loc = {}
    for zip,address,name in location_list:
        if zip not in grouped_loc: grouped_loc[zip] = []
        grouped_loc[zip].append((address,name))
    return grouped_loc



def process_items(items):
    uf = UnionFind(items)   # add the locations into UnionFind
    for i in range(len(items)):
        item1 = items[i]
        for j in range(i+1,len(items)):
            item2 = items[j]

            # check whether item1 and item2 are already in same union
            head1 = uf.find(item1)
            head2 = uf.find(item2)
            if head1 == head2:
                # if already same union, we don't need to process them again, just skip to next iteration
                continue
            
            # otherwise, check similarity, merge the items if similar
            address_sim = get_similarity(head1[0],head2[0])
            name_sim = get_similarity(head1[1],head2[1])
            if address_sim >= 0.7 and name_sim >= 0.6:     
                uf.union(item1,item2)

    
    # define a function to calculate
    def get_most_frequent(uf):
        counts = uf.counts  

        res = {}
        for head,address_list in counts.items():
            if not address_list: continue

            # every non-empty address_list represents a similar group
            name_frequency = collections.Counter()
            address_frequency = collections.Counter()

            for address,name in address_list:
                name_frequency[name]+=1
                address_frequency[address]+=1
            
            most_frequent_name = sorted(name_frequency.items(),key=lambda x:-x[1])[0][0]
            most_frequent_address = sorted(address_frequency.items(),key=lambda x:-x[1])[0][0]

            res[head] = (most_frequent_name,most_frequent_address)

        return res
    
    dic = get_most_frequent(uf)

    processed_items = []
    
    for item in items:
        head = uf.find(item)
        processed_items.append(dic[head])
    return processed_items


def run_add(location):
    location_list = location.copy()
    grouped_loc = preprocess(location_list)
    res = {}
    for zipcode,items in grouped_loc.items():
        temp =[]
        processed_items = process_items(items)
        res[(zipcode,processed_items[0][0],processed_items[0][1])] = []
        for a in items:
            temp = (zipcode,a[0],a[1])
            res[(zipcode,processed_items[0][0],processed_items[0][1])].append(temp)

    reverse_res = {}
    for key, values in res.items():
        for value in values:
            if value not in reverse_res:
                reverse_res[value]=[]
            reverse_res[value].append(key)
    # print(reverse_res)
    result_zip = []
    result_add = []
    result_name = []
    for locat in location:
        temp_zip= reverse_res.get(locat)[0][0]
        result_zip.append(temp_zip)
        temp_add= reverse_res.get(locat)[0][1]
        result_add.append(temp_add)
        temp_name= reverse_res.get(locat)[0][2]
        result_name.append(temp_name)
    # zip_l = []
    # name_l = []
    # add = []
    # for zipcode, info_list in res.items():
    #     for name, address in info_list:
    #         zip_l.append(zipcode)
    #         name_l.append(name)
    #         add.append(address)
    # return zip_l,name_l,add
    return result_zip,result_add,result_name

valid_class = [50,55,60,65,70,77.5,85,92.5,100,110,125,150,175,200,250,300,400,500]
def find_closest_valid_class(class_code,valid_class_list):
    class_code = float(class_code)
    closest_class = min(valid_class_list,key = lambda x: abs(x- class_code))
    return closest_class 

def standardize_class_code(class_codes, valid_class_list):
    closest_valid_classes = []
    for class_code in class_codes:
        if class_code in valid_class:
            closest_valid_classes.append(class_code)
        else:
            closest_valid = find_closest_valid_class(class_code, valid_class_list)
            closest_valid_classes.append(closest_valid)
    return closest_valid_classes

def match_zip(zipcodes, cities, states, countries):
    error_message = []
    for zipcode, city, state, country in zip(zipcodes, cities, states, countries):
        Corr_city, Corr_state= get_location_info(zipcode, country)
        if Corr_city is None or Corr_state is None:
            error_message.append("Zipcode Invalid")
        elif Corr_city != city.upper():
            error_message.append("Expected City: " + Corr_city)
        elif Corr_state != state.upper():
            error_message.append("Expected State" + Corr_state)
        else:
            error_message.append("")
    return error_message

def standardize_df(df):
    #Origin
    
    std_Oaddress = replace_diffseq_addresses(address_corrector(df["Origin Address"]))
    std_OState = state_corrector(df["Origin State"])
    std_OCity = city_corrector(df["Origin City"])
    std_Ocountry = country_corrector(df["Origin Country"])
    std_Ozip = correct_n_zip(std_Ocountry,df["Origin Zip"])
    #Destination
    std_DAddress = replace_diffseq_addresses(address_corrector(df["Destination Address"]))
    std_DState = state_corrector(df["Destination State"])
    std_DCity = city_corrector(df["Destination City"])
    std_Dcountry = country_corrector(df["Destination Country"])
    std_Dzip = correct_n_zip(std_Dcountry,df["Destination Zip"])
    #Zip Match
    Ozipmatch = match_zip(std_Ozip,std_OCity,std_OState,std_Ocountry)
    Dzipmatch = match_zip(std_Dzip ,std_DCity,std_DState,std_Dcountry)
    #Match similar address
    O_location_list = list(zip(std_Ozip,std_Oaddress,df["Origin Name"]))
    D_location_list = list(zip(std_Dzip,std_DAddress,df["Destination Name"]))
    O_con_zip,O_con_name,O_con_add = run_add(O_location_list)
    D_con_zip,D_con_name,D_con_add = run_add(D_location_list)
    
    df['con_Oname'] = O_con_name
    df['con_OAddress']=O_con_add
    df['std_OCity'] = std_OCity
    df['std_OState']=std_OState
    df['std_Ocountry'] = std_Ocountry
    df['std_Ozip']=std_Ozip
    df['Ozipmatch']=Ozipmatch
    df['con_Dname'] = D_con_name
    df['con_DAddress']=D_con_add
    df['std_DCity']=std_DCity
    df['std_DState']=std_DState
    df['std_Dcountry'] = std_Dcountry
    df['std_Dzip']=std_Dzip
    df['Dzipmatch']=Dzipmatch

    return df

st.title("Location Corrector")
st.header('File Input')
input_file = st.file_uploader("Upload an Excel File",type=["xls","xlsx"])
if input_file is not None:
    df = pd.read_excel(input_file, sheet_name='LTL Sample Data')
    df = df[df['ID'].notna()]
    # Get the column names
    column_names = df.columns.tolist()
    column_names_short = ['ID',
    'Origin Name',
    'Origin Address',
    'Origin City',
    'Origin State',
    'Origin Zip',
    'Origin Country',
    'OCity-State-Zip',
    'Destination Name',
    'Destination Address',
    'Destination City',
    'Destination State',
    'Destination Zip',
    'Destination Country',
    'DCity-State-Zip']
    # Create a dictionary to store the columns as lists
    df_as_lists = {}

    # Loop through each column and store its values as a list in the dictionary
    for column_name in column_names_short:
        df_as_lists[column_name] = df[column_name].tolist()
    result = standardize_df(df_as_lists)
    df_std = pd.DataFrame(result)
    st.dataframe(df_std)
    # st.header('Download output')
    # custom_filename = st.text_input("Enter the custom filename (e.g., MyCustomFile.xlsx):")
    # if st.button("Download"): 
    #     default_filename = "Cleaned Location.xlsx"

    #     # Determine the output filename
    #     if custom_filename:
    #         filename = custom_filename
    #     else:
    #         filename = default_filename
    #     output_path = filename
    #     with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    #         df_std.to_excel(writer, index=False)
