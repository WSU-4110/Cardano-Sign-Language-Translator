import unittest
from c1 import*



class TestCamera(unittest.TestCase):

    def test_camera(self):
        hit = detection()
        expected = True;
        self.assertEqual(expected, hit)

    def test_translation(self):
         expected = "hallo "
         hit = translation()
         self.assertEqual(expected, hit)

    def test_get_suggestion(self):
        hit = get_suggestion()
        expected = "my name"
        self.assertEqual(expected, hit)

    def test_validation_step(self):
        expected = "b"
        hit = validation_step()
        self.assertEqual(expected, hit)

    def test_validation_epoch_end(self):
        expected = "c"
        hit = validation_epoch_end()
        self.assertEqual(expected, hit)


    def test_epoch_end(self):
        expected = "d"
        hit = epoch_end()
        self.assertEqual(expected, hit)

    def test_predict_image(self):
        expected = "A"
        hit = predict_image()
        self.assertEqual(expected, hit, "Testing predict image")
    def test_training_step(self):
        expected = "e"
        hit = training_step()
        self.assertEqual(expected, hit)
if __name__ == '__main__':

    
    unittest.main()
    



        
    



    








