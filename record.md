[0:id, 7:neighborhood_overview, 9:host_id, 15:host_response_time, 16:host_response_rate, 17:host_acceptance_rate(Survivorship bias?), 18:host_is_superhost, 26:host_identity_verified, 28:neighbourhood_cleansed, 30:latitude, 31:longitude, 32:property_type, 33:room_type, 34:accommodates, 36:bathrooms_text, 37:bedrooms, 38:beds, 39:amenities, 40:price, 41:minimum_nights, 50:has_availability, 51:availability_30, 56:number_of_reviews, (57:number_of_reviews_ltm, 58:number_of_reviews_l30d,) 59:first_review, 69:instant_bookable, 70:calculated_host_listings_count, 74:reviews_per_month]

# Idea
Fill the blanks: use scores to train a model for target, then use this model to fill the blanks.

cut:
[](](]


## Fixed: property_type / bathrooms_text
Amount less than 100 replace to others, or not?

## Fixed: amenities
The same procedure with reviews