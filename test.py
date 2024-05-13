import lancedb
from lancedb.pydantic import LanceModel, Vector

# Connect to LanceDB or create a new database if it doesn't exist
db = lancedb.connect("./lancedb")


class NestedField(LanceModel):
    id: int


# Define the schema for the table
class SampleSchema(LanceModel):
    metadata: NestedField
    vector: Vector(2)


# Create a new table or open if it already exists
table_name = "sample_table"

db.drop_table(table_name)

if table_name in db:
    tbl = db.open_table(table_name)
else:
    tbl = db.create_table(table_name, schema=SampleSchema)

# Insert sample data into the table
data = [
    {
        "metadata": {
            "id": 10,
        },
        "vector": [0.1, 0.1],
    },
    {
        "metadata": {
            "id": 20,
        },
        "vector": [0.2, 0.2],
    },
    {
        "metadata": {
            "id": 20,
        },
        "vector": [0.3, 0.3],
    },
]
data = [SampleSchema(**row) for row in data]

tbl.add(data)


print("Filtering Enabled =======")

# Perform a search with filtering
result = tbl.search([0.5, 0.2]).where("metadata.id = 10", prefilter=True).to_list()

print(result)

print("Filtering Disabled =======")

# Perform a search with filtering
result = tbl.search([0.5, 0.2]).to_list()

print(result)
