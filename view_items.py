from flask_backend import app, db, Item

def view_items():
    with app.app_context():
        items = Item.query.all()
        if not items:
            print("No items found in the database.")
            return

        print(f"{'ID':<5} {'Name':<50} {'Description'}")
        print("-" * 80)
        for item in items:
            print(f"{item.id:<5} {item.name:<50} {item.description}")

if __name__ == "__main__":
    view_items()
