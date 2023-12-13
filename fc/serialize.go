package fc

import (
	"encoding/gob"
	"fmt"
	"os"
)

func (fc *FC) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("unable to open file. %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	err = encoder.Encode(fc)
	if err != nil {
		return fmt.Errorf("unable to encode FC struct. %v", err)
	}

	return nil
}

func Load(filePath string) (*FC, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("unable to open file. %v", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	fc := &FC{}
	err = decoder.Decode(fc)
	if err != nil {
		return nil, fmt.Errorf("unable to decode into FC struct. %v", err)
	}

	return fc, nil
}
