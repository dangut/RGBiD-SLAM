/**
* This file is part of RGBID-SLAM.
*
* Copyright (C) 2015 Daniel Gutiérrez Gómez <danielgg at unizar dot es> (Universidad de Zaragoza)
*
* RGBID-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* RGBID-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with RGBID-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SETTINGS_HPP_
#define SETTINGS_HPP_


#include <map>
#include <iostream>
#include <string>


namespace RGBID_SLAM
{ 
  inline std::string trim(std::string src, char const* delims = " \t\r\n")
  {
    std::string res(src);
    std::string::size_type index = res.find_last_not_of(delims);
    if (index != std::string::npos)
      res.erase(++index);
    
    index = res.find_first_not_of(delims);
    if(index != std::string::npos)
      res.erase(0,index);
    else
      res.erase();
    
    return res;
  };

  class Entry
  {
    public:
      Entry(std::string name="", std::string value="");
      
      std::string getName() const {return name_;}
      std::string getValue() const {return value_;}
      
      void setValue(std::string new_value)
      {
        value_ = new_value;
      };
      
    private:
      std::string name_;
      std::string value_;
  };

  class Section
  {
    public:
      Section(std::string name="");
      
      void addEntry(Entry &new_entry);
      
      bool getEntry(const std::string &entry_name, Entry &entry) const;
      
      std::string getName() const {return name_;}
      
      std::map<std::string, Entry> entries_; 
      
    private:
      
      std::string name_;   
  };

  class Settings
  {
    public:
    
      Settings(std::ifstream& filestream);
      
      void load(std::ifstream& filestream);
      
      void addSection(Section &new_section);
      
      bool getSection(const std::string &section_name, Section& section) const;
      
    private:
    
      std::map<std::string, Section> sections_; 
  };
    
};
    
#endif /* SETTINGS_HPP_ */
