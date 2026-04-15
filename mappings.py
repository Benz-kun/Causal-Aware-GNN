# File: src/mappings.py

# A simplified mapping of HS2 Codes (2-digit) to Broad Economic Sectors
# In a real paper, you would use the full 56-sector WIOD concordance table.

HS_TO_SECTOR = {
    # ENERGY & MINING
    '27': 'Energy_Mining', '26': 'Energy_Mining', '25': 'Energy_Mining',
    
    # AGRICULTURE & FOOD
    '01': 'Agriculture', '02': 'Agriculture', '03': 'Agriculture', '04': 'Agriculture',
    '07': 'Agriculture', '08': 'Agriculture', '10': 'Agriculture', '09': 'Agriculture',
    
    # TEXTILES
    '50': 'Textiles', '51': 'Textiles', '52': 'Textiles', '60': 'Textiles', 
    '61': 'Textiles', '62': 'Textiles', '63': 'Textiles',
    
    # CHEMICALS & PLASTICS
    '28': 'Chemicals', '29': 'Chemicals', '30': 'Chemicals', '31': 'Chemicals',
    '39': 'Chemicals', '40': 'Chemicals',
    
    # METALS
    '72': 'Metals', '73': 'Metals', '74': 'Metals', '76': 'Metals',
    
    # ELECTRONICS & COMPUTERS
    '85': 'Electronics',
    
    # MACHINERY
    '84': 'Machinery',
    
    # AUTOMOTIVE (TRANSPORT)
    '87': 'Automotive', '86': 'Automotive', '88': 'Automotive', '89': 'Automotive',
    
    # CONSTRUCTION MATERIALS
    '68': 'Construction_Mat', '69': 'Construction_Mat', '70': 'Construction_Mat'
}

def get_sector(hs_code):
    """Returns the Sector Name for a given HS Code (2-digit)."""
    # Clean the code (ensure it's 2 digits)
    code_str = str(hs_code).zfill(2)
    return HS_TO_SECTOR.get(code_str, 'Other_Goods')

# Map GDELT 2-letter codes to our 3-letter ISO codes
GDELT_TO_ISO = {
    'US': 'USA', 'CH': 'CHN', 'RS': 'RUS', 'GM': 'DEU', 
    'JA': 'JPN', 'FR': 'FRA', 'UK': 'GBR', 'IN': 'IND', 
    'BR': 'BRA', 'CA': 'CAN', 'MX': 'MEX', 'AS': 'AUS'
}

def get_iso3(code_2char):
    """Converts 2-char code (US) to 3-char ISO (USA). Returns None if unknown."""
    return GDELT_TO_ISO.get(str(code_2char).upper())