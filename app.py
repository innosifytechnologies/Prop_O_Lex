from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional
import uvicorn
import logging
import httpx

from hybrid_inference import HybridInferenceEngine

# Try to import config, use default if not found
try:
    from config import LOCATIONIQ_API_KEY
except ImportError:
    LOCATIONIQ_API_KEY = "pk.YOUR_API_KEY_HERE"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== INITIALIZE APP =====
app = FastAPI(
    title="🇮🇳 India Property Valuation API",
    description="Production-ready property valuation for India with circle-rate enforcement and LocationIQ geocoding",
    version="1.0.0"
)

# ===== CORS MIDDLEWARE =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD ENGINE AT STARTUP =====
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = HybridInferenceEngine(
            lr_model_path='./models/linear_model.pkl',
            xgb_model_path='./models/xgb_model.pkl',
            metadata_path='./models/metadata.json'
        )
        logger.info("✅ Engine loaded successfully")
        
        # Check LocationIQ API key
        if LOCATIONIQ_API_KEY == "pk.YOUR_API_KEY_HERE":
            logger.warning("⚠️ LocationIQ API key not configured. Edit config.py to enable geocoding.")
        else:
            logger.info("✅ LocationIQ API key configured")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        raise

# ===== ENUMS =====
class PropertyTypeEnum(str, Enum):
    APARTMENT = "residential_apartment"
    HOUSE = "independent_house"
    VILLA = "villa"
    ROW = "row_house"
    PENTHOUSE = "penthouse"
    PLOT = "residential_plot"
    COMMERCIAL = "commercial_property"

class PropertySubtypeEnum(str, Enum):
    # Apartments
    STUDIO = "studio_apt"
    BHK1 = "1bhk_apt"
    BHK2 = "2bhk_apt"
    BHK3 = "3bhk_apt"
    BHK4 = "4bhk_apt"
    DUPLEX_APT = "duplex_apt"
    HIGHRISE = "highrise_apt"
    SERVICE = "service_apt"
    AFFORDABLE = "affordable_apt"
    
    # Houses
    INDEPENDENT_FLOOR = "independent_floor"
    BUNGALOW = "bungalow"
    SINGLE_STOREY = "single_storey_house"
    MULTISTOREY = "multistorey_house"
    
    # Villas
    LUXURY_VILLA = "luxury_villa"
    DUPLEX_VILLA = "duplex_villa"
    TRIPLEX_VILLA = "triplex_villa"
    GATED_VILLA = "gated_villa"
    FARM_VILLA = "farm_villa"
    
    # Row Houses
    TOWNHOUSE = "townhouse"
    CLUSTER = "cluster_housing"
    ROW_VILLA = "row_villa"
    GATED_ROW = "gated_rowhouse"
    
    # Penthouses
    PENTHOUSE_SINGLE = "penthouse_single"
    PENTHOUSE_DUPLEX = "penthouse_duplex"
    PENTHOUSE_TERRACE = "penthouse_terrace"
    
    # Plots (Land Use Categories)
    RESIDENTIAL_PLOT = "residential_plot"
    COMMERCIAL_PLOT = "commercial_plot"
    INDUSTRIAL_PLOT = "industrial_plot"
    AGRICULTURAL_PLOT = "agricultural_plot"
    INSTITUTIONAL_PLOT = "institutional_plot"
    MIXED_USE_PLOT = "mixed_use_plot"
    
    # Commercial
    OFFICE = "office_space"
    IT_PARK = "it_techpark"
    SHOP = "shop_showroom"
    WAREHOUSE = "warehouse_godown"
    INDUSTRIAL = "industrial_shed"
    HOTEL = "hotel_resort"
    COWORKING = "coworking_space"
    MEDICAL = "medical_clinic"
    EDUCATIONAL = "educational_institution"

# ===== GEOCODING MODELS =====
class GeocodeSuggestion(BaseModel):
    lat: str
    lon: str
    display_name: str
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None

class GeocodeResponse(BaseModel):
    success: bool
    suggestions: List[GeocodeSuggestion]
    message: Optional[str] = None

# ===== REQUEST/RESPONSE MODELS =====
class PropertyRequest(BaseModel):
    latitude: float = Field(..., ge=6, le=37, description="Latitude (6-37)")
    longitude: float = Field(..., ge=68, le=98, description="Longitude (68-98)")
    city: str = Field(..., description="City name (e.g. bangalore)")
    state: str = Field(..., description="State name (e.g. karnataka)")
    pincode: str = Field(..., description="Postal code (e.g. 560001)")
    area_sqft: float = Field(..., gt=100, description="Built-up area in sqft")
    bedrooms: int = Field(..., ge=0, le=10, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, le=10, description="Number of bathrooms")
    property_type: PropertyTypeEnum = Field(..., description="Main property type")
    property_subtype: PropertySubtypeEnum = Field(..., description="Specific subtype")
    age: float = Field(..., ge=-1, le=100, description="Age in years (-1=new)")

class PropertyResponse(BaseModel):
    linear_price: float
    xgb_price: float
    hybrid_price: float
    final_price: float
    final_price_per_sqft: float
    subtype_multiplier: float
    confidence_score: float
    confidence_level: str
    price_band: str
    currency: str
    market: str

# ===== ENDPOINTS =====
@app.get("/health", tags=["System"], summary="Health Check")
def health_check():
    """Returns the service status and LocationIQ configuration state."""
    return {
        "status": "ok", 
        "service": "India Property Valuation API",
        "locationiq_configured": LOCATIONIQ_API_KEY != "pk.YOUR_API_KEY_HERE"
    }

@app.get("/geocode", response_model=GeocodeResponse, tags=["Location"], summary="Search Locations")
async def geocode_search(q: str = Query(..., min_length=3, description="Search query")):
    """
    Search for locations using LocationIQ autocomplete.
    Returns suggestions with lat, lon, city, state, pincode.
    """
    if LOCATIONIQ_API_KEY == "pk.YOUR_API_KEY_HERE":
        # Return demo data if API key not configured
        demo_suggestions = [
            GeocodeSuggestion(lat="12.9716", lon="77.5946", display_name="MG Road, Bangalore, Karnataka, India", city="Bangalore", state="Karnataka", pincode="560001"),
            GeocodeSuggestion(lat="19.0760", lon="72.8777", display_name="Bandra West, Mumbai, Maharashtra, India", city="Mumbai", state="Maharashtra", pincode="400050"),
            GeocodeSuggestion(lat="28.6139", lon="77.2090", display_name="Connaught Place, New Delhi, Delhi, India", city="New Delhi", state="Delhi", pincode="110001"),
            GeocodeSuggestion(lat="17.3850", lon="78.4867", display_name="Jubilee Hills, Hyderabad, Telangana, India", city="Hyderabad", state="Telangana", pincode="500033"),
            GeocodeSuggestion(lat="13.0827", lon="80.2707", display_name="T Nagar, Chennai, Tamil Nadu, India", city="Chennai", state="Tamil Nadu", pincode="600017"),
        ]
        filtered = [s for s in demo_suggestions if q.lower() in s.display_name.lower()]
        return GeocodeResponse(
            success=True, 
            suggestions=filtered[:5],
            message="Demo mode - configure LocationIQ API key in config.py for live search"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://api.locationiq.com/v1/autocomplete"
            params = {
                "key": LOCATIONIQ_API_KEY,
                "q": q,
                "countrycodes": "in",
                "limit": 5,
                "dedupe": 1,
                "addressdetails": 1
            }
            response = await client.get(url, params=params, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                suggestions = []
                for item in data:
                    address = item.get("address", {})
                    suggestions.append(GeocodeSuggestion(
                        lat=item.get("lat", "0"),
                        lon=item.get("lon", "0"),
                        display_name=item.get("display_name", ""),
                        city=address.get("city") or address.get("town") or address.get("village") or address.get("county"),
                        state=address.get("state"),
                        pincode=address.get("postcode")
                    ))
                return GeocodeResponse(success=True, suggestions=suggestions)
            else:
                logger.error(f"LocationIQ error: {response.status_code} - {response.text}")
                return GeocodeResponse(success=False, suggestions=[], message=f"LocationIQ error: {response.status_code}")
    
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return GeocodeResponse(success=False, suggestions=[], message=str(e))

@app.post("/predict", response_model=PropertyResponse, tags=["Valuation"], summary="Predict Property Price")
def predict(request: PropertyRequest):
    """
    High-accuracy valuation prediction using Hybrid ML (XGBoost + Linear Regression).
    Includes location features, historical trends, and property attributes.
    """
    try:
        input_dict = request.model_dump()
        logger.info(f"Prediction: {request.property_type}.{request.property_subtype} @ {request.city}, pincode={request.pincode}")
        
        result = engine.predict_single(input_dict)
        
        logger.info(f"  → Final price: ₹{result['final_price']:,.0f}, Confidence: {result['confidence_level']}")
        return PropertyResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SERVE FRONTEND =====
from fastapi.responses import FileResponse
import os

@app.get("/")
def read_root():
    return FileResponse('index.html')

# ===== RUN =====
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
