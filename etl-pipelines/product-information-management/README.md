### Bulk Staging Data Store
This is the staging ground for the data to reside in before it is tagged and shipped into the vector storage data house.  Primarily used by the Catalog Crawlers which goes to different catalogs and finds each of the new updates using hash comparison. These hashes reside here and instead of storing the whole catalog on disk only the comparable hashes are stored against each catalog and product. 
Since **Catalog Data** is fetched as a JSON object, this is converted to a string representation and then hashed.
Two major hash types will be stored
 - Catalog level hash
	 - If this matching fails, it indicates whether the crawler has to check for any product updates or additions within the catalog.
 - Catalog Product Level Hash
	 - If the above matching has failed the crawler goes down to each product and then compares the JSON string hash to the stored hash of the product to see any updates has been made.


### Product-Vendor Mapping Store
#### Case 1: Branded Product
A branded product could be sold by multiple vendors and needs to be indexed against a single entity/product within the system to prevent the same product being represented multiple times within a wide product search within the system. This is quite easy when the approached product already has a universal identifier code. This helps within the marketplace discovery as well, as a vendor could have given more key-attribute information of the same product that another vendor might have missed. 

Since they both are certifiably the same product, their attributes can be linked and create a better product listing in the marketplace aiding in better product discovery. Finally the customer can select the seller based on their ratings or stock availability of the product.

**TL;DR** Summarizing:

| Scenario | Description |
| ---- | ---- |
| Branded products can be sold by multiple vendors. | This can create duplicate entries for the same product in a product search. |
| Products need to be indexed against a single entity. | This prevents duplicate entries and ensures that all information about a product is in one place. |
| A universal identifier code can make this process easier. | A universal identifier code is recognized by UPC, EAN or a bar-code identification issued by a competent authority. |
| Attributes from different vendors can be linked to create a better product listing. | This can provide customers with more information about a product and help them make a decision about which vendor to purchase from. |
| Customers can select a vendor based on ratings or stock availability. | This gives customers more flexibility and choice when purchasing a product. |
#### Case 2: Non - Branded Product 
For a non branded product  a product data hash is created after extraction of the product features removing all identifier features and compared against the existing database for existing indexed products. Provisionally create a new indexed product store. If there was an existing product in the database in the former step notify the vendor via email about the similar product and ask to re-upload the product with modified data.

<div style="background:#edf5ff">
<img src="../../images/Product Addition Pipeline.svg" width="2048"/>
</div>