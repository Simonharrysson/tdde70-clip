from torch.utils.data import Dataset, DataLoader

class CLIPDataset(Dataset):
    def __init__(self, data_list, image_dir, tokenizer, preprocess):
        self.data = data_list
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.image_dir / item["filename"]
        
        # Load and preprocess image
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        
        # Get one caption (RSICD usually has 5, we'll take the first)
        caption = item["sentences"][0]["raw"]
        text = self.tokenizer(caption).squeeze(0)

        return image, text

# Filter for the training split
train_data = [img for img in data["images"] if img["split"] == "train"]
train_dataset = CLIPDataset(train_data, IMAGE_DIR, tokenizer, preprocess)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)